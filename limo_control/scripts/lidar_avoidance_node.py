#!/usr/bin/env python3
"""Standalone reactive exploration node for the LIMO cobot."""

from __future__ import annotations

import math
import os
import threading
from typing import List, Optional, Tuple

import rospy
import tf2_ros
import yaml
import numpy as np
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

import rospkg
from diagnostic_updater import FunctionDiagnosticTask, Updater
from dynamic_reconfigure.server import Server as DynServer

try:
    from limo_control.cfg import AvoidanceConfig
except ImportError:
    AvoidanceConfig = None
    rospy.logwarn("[lidar_avoidance] limo_control.cfg modules not found; dynamic_reconfigure disabled")

from patrol_modules.dynamic_tracker import DynamicTracker, TrackParams
from patrol_modules.frontier_explore import FrontierExplorer, FrontierGoal, FrontierParams
from patrol_modules.lidar_avoid import AvoidParams, LidarAvoider


class LidarAvoidanceNode:
    def __init__(self) -> None:
        rospy.init_node("limo_lidar_avoidance")
        self._lock = threading.RLock()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._avoid_params = self._load_params("lidar_avoidance.yaml", AvoidParams, "avoid_params")
        self._track_params = self._load_params("dynamic_tracker.yaml", TrackParams, "tracker_params")
        self._frontier_params = self._load_params("frontier_explore.yaml", FrontierParams, "frontier_params")

        self._avoider = LidarAvoider(self._avoid_params)
        self._tracker = DynamicTracker(self._track_params)
        self._explorer = FrontierExplorer(self._frontier_params)

        self._base_frame = rospy.get_param("~base_frame", "base_link")
        self._odom_frame = rospy.get_param("~odom_frame", "odom")
        self._map_frame = rospy.get_param("~map_frame", "map")

        self._scan_timeout = float(rospy.get_param("~scan_timeout", 0.5))
        self._tf_timeout = float(rospy.get_param("~tf_timeout", 0.2))

        cmd_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self._cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)
        self._debug_pub = rospy.Publisher("~avoidance_debug", String, queue_size=10)

        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        map_topic = rospy.get_param("~map_topic", "/map")

        self._scan_sub = rospy.Subscriber(scan_topic, LaserScan, self._scan_cb, queue_size=1)
        self._odom_sub = rospy.Subscriber(odom_topic, Odometry, self._odom_cb, queue_size=10)

        self._map_enabled = bool(map_topic)
        self._map_received = False
        if self._map_enabled:
            self._map_sub = rospy.Subscriber(map_topic, OccupancyGrid, self._map_cb, queue_size=1)
        else:
            self._map_sub = None

        self._last_goal: Optional[FrontierGoal] = None
        self._last_scan_time: Optional[rospy.Time] = None
        self._last_scan: Optional[Tuple[LaserScan, np.ndarray]] = None
        self._map: Optional[OccupancyGrid] = None
        self._last_cmd = Twist()
        self._last_track_count: int = 0

        self._diag = Updater()
        self._diag.setHardwareID("limo_lidar_avoidance")
        self._diag.add(FunctionDiagnosticTask("inputs", self._diag_inputs))

        self._timer = rospy.Timer(rospy.Duration(0.1), self._timer_cb)

        if AvoidanceConfig is not None:
            self._dyn_srv = DynServer(AvoidanceConfig, self._on_dyn_cfg)
        else:
            self._dyn_srv = None

        self._self_check()

    # ------------------------------------------------------------------
    def _load_params(self, filename: str, cls, param_ns: str):
        pkg_path = rospkg.RosPack().get_path("limo_control")
        default_path = os.path.join(pkg_path, "config", filename)
        param_path = rospy.get_param(f"~{filename}", default_path)

        try:
            with open(param_path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except OSError as exc:
            rospy.logfatal(f"Failed to load parameters from {param_path}: {exc}")
            raise

        if isinstance(data, dict):
            if param_ns in data and isinstance(data[param_ns], dict):
                base = dict(data[param_ns])
            else:
                base = dict(data)
        else:
            base = {}

        tree = rospy.get_param(f"~{param_ns}", None)
        if isinstance(tree, dict) and tree:
            override = tree[param_ns] if param_ns in tree and isinstance(tree[param_ns], dict) else tree
            base.update(override)

        return cls(**base)

    def _on_dyn_cfg(self, cfg, _level):
        with self._lock:
            self._avoid_params.v_max = float(cfg.v_max)
            self._avoid_params.w_max = float(cfg.w_max)
            k = max(1, int(cfg.median_window))
            if k % 2 == 0:
                k += 1
            self._avoid_params.median_window = k
            self._avoid_params.ttc_stop = float(cfg.ttc_stop)
            self._avoid_params.ttc_slow = float(cfg.ttc_slow)
            cfg.median_window = k
        return cfg

    def _scan_cb(self, msg: LaserScan) -> None:
        rng = np.asarray(msg.ranges, dtype=np.float32)
        rng[~np.isfinite(rng)] = msg.range_max
        rng = np.clip(rng, msg.range_min, msg.range_max)

        k = int(getattr(self._avoid_params, "median_window", 3))
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        self._avoid_params.median_window = k
        if k > 1:
            pad = k // 2
            if pad > 0:
                padv = np.pad(rng, (pad, pad), mode="edge")
                rng = np.array(
                    [np.median(padv[i - pad : i + pad + 1]) for i in range(pad, len(padv) - pad)],
                    dtype=np.float32,
                )

        clean_scan = LaserScan()
        clean_scan.header = msg.header
        clean_scan.angle_min = msg.angle_min
        clean_scan.angle_max = msg.angle_max
        clean_scan.angle_increment = msg.angle_increment
        clean_scan.time_increment = msg.time_increment
        clean_scan.scan_time = msg.scan_time
        clean_scan.range_min = msg.range_min
        clean_scan.range_max = msg.range_max
        clean_scan.ranges = rng.tolist()
        clean_scan.intensities = list(msg.intensities)

        with self._lock:
            self._tracker.update_scan(clean_scan)
            self._avoider.update_scan(clean_scan)
            self._last_scan_time = rospy.Time.now()
            self._last_scan = (clean_scan, rng)

    def _odom_cb(self, msg: Odometry) -> None:
        with self._lock:
            self._avoider.update_odom(msg)
            self._update_robot_pose()

    def _map_cb(self, msg: OccupancyGrid) -> None:
        with self._lock:
            self._explorer.update_map(msg)
            self._map_received = True
            self._map = msg

    def _update_robot_pose(self) -> None:
        if not self._map_enabled or not self._map_received:
            return
        try:
            trans = self._tf_buffer.lookup_transform(
                self._map_frame,
                self._base_frame,
                rospy.Time(0),
                rospy.Duration(self._tf_timeout),
            )
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            quat = trans.transform.rotation
            yaw = self._yaw_from_quaternion(quat.x, quat.y, quat.z, quat.w)
            self._explorer.set_robot_pose(x, y, yaw)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

    def _timer_cb(self, _event) -> None:
        with self._lock:
            now = rospy.Time.now()

            if self._last_scan_time is None or (now - self._last_scan_time) > rospy.Duration(self._scan_timeout):
                rospy.logwarn_throttle(1.0, "LiDAR scan timeout; stopping robot")
                self._avoider.update_nav_hint(None)
                zero = Twist()
                self._cmd_pub.publish(zero)
                self._last_cmd = zero
                self._last_goal = None
                self._last_track_count = 0
                self._diag.update()
                return

            tracked = self._tracker.step(now.to_sec())
            self._last_track_count = len(tracked)
            barriers = self._convert_barriers(tracked)
            self._avoider.ingest_dynamic_barriers(barriers)

            goal: Optional[FrontierGoal] = None
            if self._map_enabled and self._map_received:
                goal = self._explorer.pick_next_goal()

            if goal is not None:
                self._set_nav_hint(goal)
                self._last_goal = goal
            elif self._map_enabled and self._map_received and self._last_goal is not None:
                self._set_nav_hint(self._last_goal)
            else:
                self._avoider.update_nav_hint(None)
                if not self._map_received:
                    self._last_goal = None

            cmd, debug = self._avoider.compute_cmd()
            self._cmd_pub.publish(cmd)
            self._last_cmd = cmd
            if self._avoid_params.publish_debug:
                self._debug_pub.publish(self._avoider.to_json(debug))

            self._diag.update()

    def _convert_barriers(self, tracks) -> List[dict]:
        barriers: List[dict] = []
        robot_speed = self._avoider.current_speed()
        for obj in tracks:
            if not obj.is_dynamic:
                continue
            radial_dir = (math.cos(obj.theta), math.sin(obj.theta))
            v_rel = obj.vx * radial_dir[0] + obj.vy * radial_dir[1] - robot_speed
            radius = max(
                self._avoid_params.proxemics_min,
                self._avoid_params.dyn_inflation_base + self._avoid_params.dyn_inflation_gain * obj.speed,
            )
            barriers.append(
                {
                    "theta": obj.theta,
                    "range": obj.range,
                    "v_rel": v_rel,
                    "radius": radius,
                }
            )
        return barriers

    def _set_nav_hint(self, goal: FrontierGoal) -> None:
        heading = goal.heading_hint
        hint = Vector3(x=math.cos(heading), y=math.sin(heading), z=0.0)
        self._avoider.update_nav_hint(hint)

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        t0 = +2.0 * (w * z + x * y)
        t1 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t0, t1)

    def _diag_inputs(self, stat):
        now = rospy.Time.now()
        if self._last_scan_time is None:
            age = float("inf")
        else:
            age = (now - self._last_scan_time).to_sec()
        if age < self._scan_timeout:
            stat.summary(0, "OK")
        else:
            stat.summary(1, "No recent scan")
        stat.add("scan_age_sec", age)
        stat.add("map_available", bool(self._map))
        stat.add("tf_timeout_s", getattr(self, "_tf_timeout", 0.2))
        stat.add("tracked_objects", self._last_track_count)
        stat.add("cmd_linear_x", self._last_cmd.linear.x)
        stat.add("cmd_angular_z", self._last_cmd.angular.z)
        return stat

    def _self_check(self) -> None:
        strict = bool(rospy.get_param("~strict_self_check", False))
        missing: List[str] = []

        for _ in range(10):
            try:
                pubs = dict(rospy.get_published_topics())
            except Exception:
                pubs = {}

            missing = []
            for tparam, default in (("~scan_topic", "/scan"), ("~odom_topic", "/odom")):
                tname = rospy.get_param(tparam, default)
                if tname not in pubs:
                    missing.append(tname)

            if not missing:
                break

            rospy.sleep(0.5)

        tf_ok = True
        try:
            self._tf_buffer.lookup_transform(
                self._odom_frame,
                self._base_frame,
                rospy.Time(0),
                rospy.Duration(self._tf_timeout),
            )
        except Exception:
            tf_ok = False

        if missing or not tf_ok:
            rospy.logerr(
                "Self-check failed. missing_topics=%s tf_ok=%s", missing, tf_ok
            )
            if strict:
                rospy.signal_shutdown("Bringup failed")


def main() -> None:
    node = LidarAvoidanceNode()
    rospy.on_shutdown(lambda: node._cmd_pub.publish(Twist()))
    rospy.loginfo("Lidar avoidance node started")
    rospy.spin()


if __name__ == "__main__":
    main()
