#!/usr/bin/env python3
"""Standalone reactive exploration node for the LIMO cobot."""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
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


@dataclass
class ModeParams:
    refine_unknown: float
    complete_unknown: float
    complete_hold_time: float
    no_frontier_timeout: float
    spin_duration: float
    spin_speed: float
    unknown_lpf_alpha: float
    wall_follow_duration: float
    wall_follow_speed: float
    wall_follow_target_dist: float
    wait_dyn_dist: float
    wait_dyn_time: float
    return_home_tol: float


class LidarAvoidanceNode:
    def __init__(self) -> None:
        rospy.init_node("limo_lidar_avoidance")
        self._lock = threading.RLock()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._avoid_params = self._load_params("lidar_avoidance.yaml", AvoidParams, "avoid_params")
        self._track_params = self._load_params("dynamic_tracker.yaml", TrackParams, "tracker_params")
        self._frontier_params = self._load_params("frontier_explore.yaml", FrontierParams, "frontier_params")
        self._mode_params = self._load_params("explore_mode.yaml", ModeParams, "mode_params")

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
        self._unknown_ratio: float = 1.0
        self._mode: str = "NO_MAP"
        self._complete_since: Optional[rospy.Time] = None
        self._spin_until: Optional[rospy.Time] = None
        self._spin_sign: int = 1
        self._no_frontier_since: Optional[rospy.Time] = None
        self._last_frontier_time: Optional[rospy.Time] = None
        self._wait_until: Optional[rospy.Time] = None
        self._wall_follow_until: Optional[rospy.Time] = None
        self._home_pose: Optional[Tuple[float, float]] = None
        self._spin_completed_no_frontier: bool = False

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
            self._tracker.update_odom(msg)
            self._update_robot_pose()

    def _map_cb(self, msg: OccupancyGrid) -> None:
        with self._lock:
            self._explorer.update_map(msg)
            self._map_received = True
            self._map = msg
            self._update_unknown_ratio(msg)

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
            if self._home_pose is None:
                self._home_pose = (x, y)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

    def _update_unknown_ratio(self, grid: OccupancyGrid) -> None:
        data = np.asarray(grid.data, dtype=np.int8)
        total = data.size
        if total == 0:
            return
        unknown = float(np.count_nonzero(data == -1))
        ratio = unknown / float(total)
        alpha = max(0.0, min(1.0, self._mode_params.unknown_lpf_alpha))
        self._unknown_ratio = alpha * ratio + (1.0 - alpha) * self._unknown_ratio

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

            if self._maybe_wait(tracked, now):
                zero = Twist()
                self._cmd_pub.publish(zero)
                self._last_cmd = zero
                self._diag.update()
                return

            goal: Optional[FrontierGoal] = None
            map_ready = self._map_enabled and self._map_received
            if map_ready:
                goal = self._explorer.pick_next_goal()
                if goal is not None:
                    self._last_frontier_time = now

            frontier_available = goal is not None
            if map_ready:
                self._mode = self._select_mode(now, frontier_available)
            else:
                self._mode = "NO_MAP"

            # Wall-follow keeps running until時間到或發現前沿
            if self._mode == "WALL_FOLLOW":
                if self._wall_follow_until is None or now > self._wall_follow_until:
                    self._mode = "SPIN_SEARCH"
                elif frontier_available:
                    self._mode = "EXPLORE" if self._unknown_ratio > self._mode_params.refine_unknown else "REFINE"
                else:
                    cmd = self._wall_follow_cmd()
                    self._cmd_pub.publish(cmd)
                    self._last_cmd = cmd
                    self._diag.update()
                    return

            # Return home: drive toward home pose using avoider
            if self._mode == "RETURN_HOME":
                if self._home_pose is None:
                    self._mode = "COMPLETE"
                else:
                    hx, hy = self._home_pose
                    robot_pose = getattr(self._explorer, "_robot_pose", None)
                    if robot_pose is not None:
                        rx, ry, _ = robot_pose
                        dist_home = math.hypot(hx - rx, hy - ry)
                        if dist_home <= self._mode_params.return_home_tol:
                            self._mode = "COMPLETE"
                        else:
                            heading = math.atan2(hy - ry, hx - rx)
                            self._avoider.update_nav_hint(Vector3(x=math.cos(heading), y=math.sin(heading), z=0.0))
                            cmd, debug = self._avoider.compute_cmd()
                            self._cmd_pub.publish(cmd)
                            self._last_cmd = cmd
                            if self._avoid_params.publish_debug:
                                self._debug_pub.publish(self._avoider.to_json(debug))
                            self._diag.update()
                            return

            if self._mode == "COMPLETE":
                self._avoider.update_nav_hint(None)
                zero = Twist()
                self._cmd_pub.publish(zero)
                self._last_cmd = zero
                self._diag.update()
                return

            if self._mode == "SPIN_SEARCH":
                self._avoider.update_nav_hint(None)
                spin_cmd = Twist()
                spin_cmd.angular.z = self._mode_params.spin_speed * float(self._spin_sign)
                self._cmd_pub.publish(spin_cmd)
                self._last_cmd = spin_cmd
                self._diag.update()
                return

            if goal is not None:
                self._set_nav_hint(goal)
                self._last_goal = goal
            elif self._map_enabled and self._map_received and self._last_goal is not None and self._mode != "NO_MAP":
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
            speed_rel = getattr(obj, "speed_rel", obj.speed)
            radial_dir = (math.cos(obj.theta), math.sin(obj.theta))
            v_rel = obj.vx * radial_dir[0] + obj.vy * radial_dir[1] - robot_speed
            radius = max(
                self._avoid_params.proxemics_min,
                self._avoid_params.dyn_inflation_base + self._avoid_params.dyn_inflation_gain * speed_rel,
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

    def _maybe_wait(self, tracks, now: rospy.Time) -> bool:
        # Enter wait if dynamic obstacle is very close
        if self._wait_until is not None and now < self._wait_until:
            self._mode = "WAIT"
            return True
        for obj in tracks:
            if not obj.is_dynamic:
                continue
            if obj.range <= self._mode_params.wait_dyn_dist:
                self._wait_until = now + rospy.Duration(self._mode_params.wait_dyn_time)
                self._mode = "WAIT"
                return True
        self._wait_until = None
        return False

    def _wall_follow_cmd(self) -> Twist:
        cmd = Twist()
        if self._last_scan is None:
            return cmd
        scan, rng = self._last_scan
        angles = scan.angle_min + np.arange(len(rng)) * scan.angle_increment
        window = math.radians(15.0)
        left_mask = (angles > math.pi / 2 - window) & (angles < math.pi / 2 + window)
        right_mask = (angles < -math.pi / 2 + window) & (angles > -math.pi / 2 - window)
        left_d = np.median(rng[left_mask]) if np.any(left_mask) else float("inf")
        right_d = np.median(rng[right_mask]) if np.any(right_mask) else float("inf")
        side = -1 if right_d <= left_d else 1  # -1: right-wall, +1: left-wall
        wall_dist = right_d if side == -1 else left_d
        if not math.isfinite(wall_dist):
            wall_dist = self._mode_params.wall_follow_target_dist * 2.0
        err = self._mode_params.wall_follow_target_dist - wall_dist
        yaw_cmd = max(-self._avoid_params.w_max, min(self._avoid_params.w_max, 1.0 * err * side))
        # Slow down if front too close
        front_min = float(np.min(rng)) if rng.size > 0 else float("inf")
        v = self._mode_params.wall_follow_speed
        if front_min < self._avoid_params.stop_distance * 1.2:
            v = 0.0
        cmd.linear.x = v
        cmd.angular.z = yaw_cmd
        return cmd

    def _select_mode(self, now: rospy.Time, frontier_available: bool) -> str:
        # Keep spinning until time window ends
        if self._mode == "SPIN_SEARCH" and self._spin_until is not None:
            if now < self._spin_until:
                return "SPIN_SEARCH"
            # Spin finished
            self._spin_until = None
            self._spin_completed_no_frontier = not frontier_available

        # Map completion check → return home
        if self._unknown_ratio <= self._mode_params.complete_unknown and not frontier_available:
            if self._complete_since is None:
                self._complete_since = now
            elif (now - self._complete_since).to_sec() >= self._mode_params.complete_hold_time:
                return "RETURN_HOME"
        else:
            self._complete_since = None

        # Trigger spin search if no frontier for a while
        if not frontier_available:
            if self._no_frontier_since is None:
                self._no_frontier_since = now
            elif (now - self._no_frontier_since).to_sec() >= self._mode_params.no_frontier_timeout:
                self._spin_until = now + rospy.Duration(self._mode_params.spin_duration)
                self._spin_sign *= -1
                self._no_frontier_since = None
                self._spin_completed_no_frontier = False
                return "SPIN_SEARCH"
        else:
            self._no_frontier_since = None
            self._spin_completed_no_frontier = False

        # After spin with no frontier → wall follow
        if not frontier_available and self._spin_completed_no_frontier:
            self._wall_follow_until = now + rospy.Duration(self._mode_params.wall_follow_duration)
            self._spin_completed_no_frontier = False
            return "WALL_FOLLOW"

        if self._unknown_ratio <= self._mode_params.refine_unknown:
            return "REFINE"
        return "EXPLORE"

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
        stat.add("mode", getattr(self, "_mode", "unknown"))
        stat.add("map_unknown_ratio", getattr(self, "_unknown_ratio", 1.0))
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
