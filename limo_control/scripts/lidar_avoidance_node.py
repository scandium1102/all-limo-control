#!/usr/bin/env python3
"""Standalone node that performs LiDAR based avoidance and exploration."""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import rospy
import rospkg
import yaml
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String

from patrol_modules import (
    AvoidParams,
    DebugInfo,
    DynamicTracker,
    FrontierExplorer,
    FrontierGoal,
    FrontierParams,
    LidarAvoider,
    TrackParams,
    TrackedObject,
)


class LidarAvoidanceNode:
    def __init__(self) -> None:
        rospy.init_node("lidar_avoidance", anonymous=False)
        pkg_path = rospkg.RosPack().get_path("limo_control")
        config_dir = rospy.get_param("~config_dir", os.path.join(pkg_path, "config"))
        avoid_cfg = self._load_yaml(os.path.join(config_dir, "lidar_avoidance.yaml"))
        tracker_cfg = self._load_yaml(os.path.join(config_dir, "dynamic_tracker.yaml"))
        frontier_cfg = self._load_yaml(os.path.join(config_dir, "frontier_explore.yaml"))

        self.avoider = LidarAvoider(AvoidParams(**avoid_cfg))
        self.tracker = DynamicTracker(TrackParams(**tracker_cfg))
        self.explorer = FrontierExplorer(FrontierParams(**frontier_cfg))

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.debug_pub = rospy.Publisher("/avoidance_debug", String, queue_size=10)

        self.control_rate = rospy.get_param("~control_rate", 20.0)
        self._timer = rospy.Timer(rospy.Duration(1.0 / max(1e-3, self.control_rate)), self._control_cb)

        rospy.Subscriber("/scan", LaserScan, self._scan_cb, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber("/map", OccupancyGrid, self._map_cb, queue_size=1)
        rospy.Subscriber("/external_emergency_stop", Bool, self._emergency_cb, queue_size=1)

        rospy.loginfo("lidar_avoidance node ready. config_dir=%s", config_dir)

    # ------------------------------------------------------------------
    def _load_yaml(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data

    # ------------------------------------------------------------------
    def _scan_cb(self, scan: LaserScan) -> None:
        self.tracker.update_scan(scan)
        self.avoider.update_scan(scan)

    # ------------------------------------------------------------------
    def _odom_cb(self, odom: Odometry) -> None:
        self.avoider.update_odom(odom)
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        q = odom.pose.pose.orientation
        yaw = self._quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.explorer.set_robot_pose(x, y, yaw)

    # ------------------------------------------------------------------
    def _map_cb(self, grid: OccupancyGrid) -> None:
        self.explorer.update_map(grid)

    # ------------------------------------------------------------------
    def _emergency_cb(self, msg: Bool) -> None:
        self.avoider.set_external_emergency_stop(msg.data)

    # ------------------------------------------------------------------
    def _control_cb(self, _: rospy.TimerEvent) -> None:
        now = rospy.get_time()
        dyn_objs = self.tracker.step(now)
        self.avoider.ingest_dynamic_barriers([self._to_barrier(obj) for obj in dyn_objs])

        goal = self.explorer.pick_next_goal()
        if goal is not None:
            hint_vec = Vector3()
            hint_vec.x = math.cos(goal.heading_hint)
            hint_vec.y = math.sin(goal.heading_hint)
            self.avoider.update_nav_hint(hint_vec)
        else:
            self.avoider.update_nav_hint(None)

        cmd, debug = self.avoider.compute_cmd()
        self.cmd_pub.publish(cmd)

        if self.avoider.params.publish_debug:
            msg = String()
            msg.data = self._serialize_debug(debug, goal, dyn_objs)
            self.debug_pub.publish(msg)

    # ------------------------------------------------------------------
    def _serialize_debug(
        self,
        debug: DebugInfo,
        goal: Optional[FrontierGoal],
        dyn_objs: list[TrackedObject],
    ) -> str:
        data = {
            "state": debug.state,
            "d_min": debug.d_min,
            "theta": debug.chosen_theta,
            "v": debug.v_cmd,
            "w": debug.w_cmd,
            "ttc_min": debug.ttc_min,
            "stuck": debug.stuck_flag,
            "notes": debug.notes,
            "gap": {
                "theta_center": getattr(debug.chosen_gap, "theta_center", None),
                "theta_best": getattr(debug.chosen_gap, "theta_best", None),
                "width": getattr(debug.chosen_gap, "width", None),
                "d_min": getattr(debug.chosen_gap, "d_min", None),
            }
            if debug.chosen_gap
            else None,
            "goal": {
                "x": goal.map_xy[0],
                "y": goal.map_xy[1],
                "heading": goal.heading_hint,
                "score": goal.score,
            }
            if goal
            else None,
            "dyn_objects": [
                {
                    "id": obj.id,
                    "range": obj.range,
                    "theta": obj.theta,
                    "vx": obj.vx,
                    "vy": obj.vy,
                    "speed": obj.speed,
                    "is_dynamic": obj.is_dynamic,
                }
                for obj in dyn_objs
            ],
        }
        return json.dumps(data, ensure_ascii=False)

    # ------------------------------------------------------------------
    def _to_barrier(self, obj: TrackedObject) -> dict:
        speed = math.hypot(obj.vx, obj.vy)
        radius = self.avoider.params.dyn_inflation_base + self.avoider.params.dyn_inflation_gain * speed
        if obj.is_dynamic:
            radius = max(radius, self.avoider.params.proxemics_min)
        else:
            radius = max(radius, self.avoider.params.base_radius)
        v_along = obj.vx * math.cos(obj.theta) + obj.vy * math.sin(obj.theta)
        v_rel = v_along - self.avoider.robot_speed
        return {"theta": obj.theta, "range": obj.range, "v_rel": v_rel, "radius": radius}

    # ------------------------------------------------------------------
    @staticmethod
    def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main() -> None:
    node = LidarAvoidanceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
