#!/usr/bin/env python3
"""Standalone reactive exploration node for the LIMO cobot."""

from __future__ import annotations

import math
import os
import threading
from typing import List, Optional

import rospkg
import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from patrol_modules.dynamic_tracker import DynamicTracker, TrackParams
from patrol_modules.frontier_explore import FrontierExplorer, FrontierGoal, FrontierParams
from patrol_modules.lidar_avoid import AvoidParams, LidarAvoider


class LidarAvoidanceNode:
    def __init__(self) -> None:
        rospy.init_node("lidar_avoidance_node")
        self._lock = threading.RLock()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._avoid_params = self._load_params("lidar_avoidance.yaml", AvoidParams)
        self._track_params = self._load_params("dynamic_tracker.yaml", TrackParams)
        self._frontier_params = self._load_params("frontier_explore.yaml", FrontierParams)

        self._avoider = LidarAvoider(self._avoid_params)
        self._tracker = DynamicTracker(self._track_params)
        self._explorer = FrontierExplorer(self._frontier_params)

        self._base_frame = rospy.get_param("~base_frame", "base_link")
        self._odom_frame = rospy.get_param("~odom_frame", "odom")
        self._map_frame = rospy.get_param("~map_frame", "map")

        self._cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self._debug_pub = rospy.Publisher("avoidance_debug", String, queue_size=10)

        self._scan_sub = rospy.Subscriber("scan", LaserScan, self._scan_cb, queue_size=1)
        self._odom_sub = rospy.Subscriber("odom", Odometry, self._odom_cb, queue_size=10)
        self._map_sub = rospy.Subscriber("map", OccupancyGrid, self._map_cb, queue_size=1)

        self._last_goal: Optional[FrontierGoal] = None
        self._timer = rospy.Timer(rospy.Duration(0.1), self._timer_cb)

    # ------------------------------------------------------------------
    def _load_params(self, filename: str, cls):
        pkg_path = rospkg.RosPack().get_path("limo_control")
        default_path = os.path.join(pkg_path, "config", filename)
        param_path = rospy.get_param(f"~{filename}", default_path)
        with open(param_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(**data)

    def _scan_cb(self, msg: LaserScan) -> None:
        with self._lock:
            self._tracker.update_scan(msg)
            self._avoider.update_scan(msg)

    def _odom_cb(self, msg: Odometry) -> None:
        with self._lock:
            self._avoider.update_odom(msg)
            self._update_robot_pose()

    def _map_cb(self, msg: OccupancyGrid) -> None:
        with self._lock:
            self._explorer.update_map(msg)

    def _update_robot_pose(self) -> None:
        try:
            trans = self._tf_buffer.lookup_transform(self._map_frame, self._base_frame, rospy.Time(0), rospy.Duration(0.05))
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            quat = trans.transform.rotation
            yaw = self._yaw_from_quaternion(quat.x, quat.y, quat.z, quat.w)
            self._explorer.set_robot_pose(x, y, yaw)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

    def _timer_cb(self, _event) -> None:
        with self._lock:
            now = rospy.Time.now().to_sec()
            tracked = self._tracker.step(now)
            barriers = self._convert_barriers(tracked)
            self._avoider.ingest_dynamic_barriers(barriers)

            goal = self._explorer.pick_next_goal()
            if goal is not None:
                self._set_nav_hint(goal)
                self._last_goal = goal
            elif self._last_goal is not None:
                self._set_nav_hint(self._last_goal)
            else:
                self._avoider.update_nav_hint(None)

            cmd, debug = self._avoider.compute_cmd()
            self._cmd_pub.publish(cmd)
            if self._avoid_params.publish_debug:
                self._debug_pub.publish(self._avoider.to_json(debug))

    def _convert_barriers(self, tracks) -> List[dict]:
        barriers: List[dict] = []
        robot_speed = self._avoider.current_speed()
        for obj in tracks:
            if not obj.is_dynamic:
                continue
            radial_dir = (math.cos(obj.theta), math.sin(obj.theta))
            v_rel = obj.vx * radial_dir[0] + obj.vy * radial_dir[1] - robot_speed
            radius = max(self._avoid_params.proxemics_min, self._avoid_params.dyn_inflation_base + self._avoid_params.dyn_inflation_gain * obj.speed)
            barriers.append({
                "theta": obj.theta,
                "range": obj.range,
                "v_rel": v_rel,
                "radius": radius,
            })
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


def main() -> None:
    node = LidarAvoidanceNode()
    rospy.loginfo("Lidar avoidance node started")
    rospy.spin()


if __name__ == "__main__":
    main()
