#!/usr/bin/env python3
"""Standalone node running LiDAR avoidance, dynamic tracking and frontier exploration."""
from __future__ import annotations

import json
import math
import os
from typing import List

import rospy
import tf.transformations as tft
import yaml
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from patrol_modules import (
    AvoidParams,
    DynamicTracker,
    FrontierExplorer,
    FrontierParams,
    LidarAvoider,
    TrackParams,
    TrackedObject,
)


class LidarAvoidanceNode:
    def __init__(self) -> None:
        rospy.init_node("lidar_avoidance_node")
        self.rate_hz = rospy.get_param("~rate", 15.0)

        self.avoid_params = AvoidParams(**self._load_yaml("config/lidar_avoidance.yaml"))
        self.track_params = TrackParams(**self._load_yaml("config/dynamic_tracker.yaml"))
        self.frontier_params = FrontierParams(**self._load_yaml("config/frontier_explore.yaml"))

        self.avoider = LidarAvoider(self.avoid_params)
        self.tracker = DynamicTracker(self.track_params)
        self.frontier = FrontierExplorer(self.frontier_params)

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.debug_pub = rospy.Publisher("/avoidance_debug", String, queue_size=10)

        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self._scan_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self._odom_cb, queue_size=1)
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self._map_cb, queue_size=1)
        self.stop_sub = rospy.Subscriber("~external_stop", Bool, self._stop_cb, queue_size=1)

        self._frontier_goal = None
        self._last_frontier_check = 0.0
        self._current_pose = (0.0, 0.0, 0.0)
        self._current_twist = None

    # ------------------------------------------------------------------
    def spin(self) -> None:
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            now = rospy.get_time()
            dyn_objs = self.tracker.step(now)
            dyn_barriers = self._convert_tracks(dyn_objs)
            self.avoider.ingest_dynamic_barriers(dyn_barriers)

            self._maybe_update_frontier(now)

            if self._frontier_goal is not None:
                hint = Vector3(
                    x=math.cos(self._frontier_goal.heading_hint),
                    y=math.sin(self._frontier_goal.heading_hint),
                    z=0.0,
                )
                self.avoider.update_nav_hint(hint)
            else:
                self.avoider.update_nav_hint(None)

            cmd, debug = self.avoider.compute_cmd()
            self.cmd_pub.publish(cmd)

            if self.avoid_params.publish_debug:
                self.debug_pub.publish(String(data=self._serialize_debug(debug)))

            if self.avoider.is_stuck():
                rospy.logwarn_throttle(5.0, "LidarAvoider reports stuck condition")

            rate.sleep()

    # ------------------------------------------------------------------
    def _scan_cb(self, msg: LaserScan) -> None:
        self.tracker.update_scan(msg)
        self.avoider.update_scan(msg)

    # ------------------------------------------------------------------
    def _odom_cb(self, msg: Odometry) -> None:
        self.avoider.update_odom(msg)
        yaw = self._yaw_from_quat(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        self._current_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw,
        )
        self._current_twist = msg.twist.twist
        self.frontier.set_robot_pose(*self._current_pose)

    # ------------------------------------------------------------------
    def _map_cb(self, msg: OccupancyGrid) -> None:
        self.frontier.update_map(msg)

    # ------------------------------------------------------------------
    def _stop_cb(self, msg: Bool) -> None:
        self.avoider.set_external_emergency_stop(bool(msg.data))

    # ------------------------------------------------------------------
    def _maybe_update_frontier(self, now: float) -> None:
        if now - self._last_frontier_check < self.frontier_params.replan_period:
            return
        goal = self.frontier.pick_next_goal()
        self._last_frontier_check = now
        if goal is not None:
            self._frontier_goal = goal
        else:
            self._frontier_goal = None

    # ------------------------------------------------------------------
    def _convert_tracks(self, tracks: List[TrackedObject]) -> List[dict]:
        barriers: List[dict] = []
        robot_speed = 0.0
        if self._current_twist is not None:
            robot_speed = self._current_twist.linear.x
        for obj in tracks:
            if not obj.is_dynamic:
                continue
            dir_x = math.cos(obj.theta)
            dir_y = math.sin(obj.theta)
            proj = obj.vx * dir_x + obj.vy * dir_y
            closing = max(0.0, robot_speed - proj)
            radius = self.avoid_params.dyn_inflation_base + self.avoid_params.dyn_inflation_gain * obj.speed
            radius = max(radius, self.avoid_params.proxemics_min)
            if closing <= 0.0 and obj.speed <= self.track_params.speed_thresh_moving:
                continue
            barriers.append({
                "theta": obj.theta,
                "range": obj.range,
                "v_rel": closing if closing > 1e-3 else obj.speed,
                "radius": radius,
            })
        return barriers

    # ------------------------------------------------------------------
    def _serialize_debug(self, debug) -> str:
        return json.dumps(
            {
                "state": debug.state,
                "d_min": debug.d_min,
                "theta": debug.chosen_theta,
                "gap": {
                    "theta_center": debug.chosen_gap.theta_center,
                    "theta_best": debug.chosen_gap.theta_best,
                    "width": debug.chosen_gap.width,
                    "d_min": debug.chosen_gap.d_min,
                },
                "v_cmd": debug.v_cmd,
                "w_cmd": debug.w_cmd,
                "ttc_min": debug.ttc_min,
                "stuck": debug.stuck_flag,
                "notes": debug.notes,
            },
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    def _load_yaml(self, relative_path: str) -> dict:
        pkg_path = rospy.get_param("~pkg_path", default=self._default_pkg_path())
        yaml_path = os.path.join(pkg_path, relative_path)
        with open(yaml_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data

    # ------------------------------------------------------------------
    @staticmethod
    def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
        _, _, yaw = tft.euler_from_quaternion((x, y, z, w))
        return yaw

    # ------------------------------------------------------------------
    @staticmethod
    def _default_pkg_path() -> str:
        import rospkg

        return rospkg.RosPack().get_path("limo_control")


def main() -> None:
    node = LidarAvoidanceNode()
    try:
        node.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
