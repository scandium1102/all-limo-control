#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone ROS node that wires the avoidance, tracking and exploration modules."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import List, Optional

import rospy
import rospkg
import yaml
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from patrol_modules.dynamic_tracker import DynamicTracker, TrackParams, TrackedObject
from patrol_modules.frontier_explore import FrontierExplorer, FrontierGoal, FrontierParams
from patrol_modules.lidar_avoid import AvoidParams, DebugInfo, LidarAvoider


class LidarAvoidanceNode:
    def __init__(self) -> None:
        rospy.init_node("lidar_avoidance_node", anonymous=False)

        pkg_path = rospkg.RosPack().get_path("limo_control")
        avoid_cfg = self._load_yaml(pkg_path + "/config/lidar_avoidance.yaml")
        track_cfg = self._load_yaml(pkg_path + "/config/dynamic_tracker.yaml")
        frontier_cfg = self._load_yaml(pkg_path + "/config/frontier_explore.yaml")

        self.avoider = LidarAvoider(AvoidParams(**avoid_cfg))
        self.tracker = DynamicTracker(TrackParams(**track_cfg))
        self.explorer = FrontierExplorer(FrontierParams(**frontier_cfg))

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.debug_pub = rospy.Publisher("/avoidance_debug", String, queue_size=10)

        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self._scan_cb, queue_size=10)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self._odom_cb, queue_size=10)
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self._map_cb, queue_size=1)
        self.estop_sub = rospy.Subscriber(
            "external_emergency_stop", Bool, self._estop_cb, queue_size=1
        )

        self._robot_speed = 0.0
        self._robot_pose = (0.0, 0.0, 0.0)
        self._last_debug: Optional[DebugInfo] = None

        rate = rospy.get_param("~control_rate", 15.0)
        self._timer = rospy.Timer(rospy.Duration(1.0 / rate), self._timer_cb)

    # ------------------------------------------------------------------
    def _load_yaml(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    # ------------------------------------------------------------------
    def _scan_cb(self, msg: LaserScan) -> None:
        self.avoider.update_scan(msg)
        self.tracker.update_scan(msg)

    def _odom_cb(self, msg: Odometry) -> None:
        self.avoider.update_odom(msg)
        pose = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self._robot_speed = msg.twist.twist.linear.x
        self._robot_pose = (pose.x, pose.y, yaw)
        self.explorer.set_robot_pose(pose.x, pose.y, yaw)

    def _map_cb(self, msg: OccupancyGrid) -> None:
        self.explorer.update_map(msg)

    def _estop_cb(self, msg: Bool) -> None:
        self.avoider.set_external_emergency_stop(msg.data)

    # ------------------------------------------------------------------
    def _timer_cb(self, event) -> None:
        now = event.current_real.to_sec() if event is not None else rospy.Time.now().to_sec()
        dyn_objs = self.tracker.step(now)
        dyn_barriers = self._convert_tracks(dyn_objs)
        self.avoider.ingest_dynamic_barriers(dyn_barriers)

        goal = self.explorer.pick_next_goal()
        hint = self._goal_to_hint(goal)
        self.avoider.update_nav_hint(hint)

        cmd, debug = self.avoider.compute_cmd()
        self.cmd_pub.publish(cmd)
        self._last_debug = debug

        if self.avoider.params.publish_debug and debug is not None:
            msg = String()
            msg.data = json.dumps(asdict(debug), ensure_ascii=False)
            self.debug_pub.publish(msg)

    # ------------------------------------------------------------------
    def _goal_to_hint(self, goal: Optional[FrontierGoal]) -> Optional[Vector3]:
        if goal is None:
            return None
        robot_x, robot_y, _ = self._robot_pose
        dx = goal.map_xy[0] - robot_x
        dy = goal.map_xy[1] - robot_y
        hint = Vector3()
        hint.x = dx
        hint.y = dy
        hint.z = 0.0
        return hint

    def _convert_tracks(self, tracks: List[TrackedObject]) -> List[dict]:
        barriers: List[dict] = []
        params = self.avoider.params
        for obj in tracks:
            radius = params.dyn_inflation_base + params.dyn_inflation_gain * obj.speed
            if obj.is_dynamic:
                radius = max(radius, params.proxemics_min)
            heading = obj.theta
            vx = obj.vx
            vy = obj.vy
            beam_dir_x = math.cos(heading)
            beam_dir_y = math.sin(heading)
            obj_speed_along = vx * beam_dir_x + vy * beam_dir_y
            v_rel = obj_speed_along - self._robot_speed
            barriers.append(
                {
                    "theta": heading,
                    "range": obj.range,
                    "v_rel": v_rel,
                    "radius": radius,
                }
            )
        return barriers


def main() -> None:
    LidarAvoidanceNode()
    rospy.loginfo("lidar_avoidance_node started")
    rospy.spin()


if __name__ == "__main__":
    main()

