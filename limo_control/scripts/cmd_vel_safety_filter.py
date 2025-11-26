#!/usr/bin/env python3
"""
cmd_vel_safety_filter.py

Pass-through safety layer for move_base/explore_lite output. It forwards the
incoming cmd_vel unless a LiDAR stop/slow condition is met, in which case it
zeros the command. Intended as a lightweight "insurance" layer when using
explore_lite+move_base.
"""

import math
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class CmdVelSafety:
    def __init__(self):
        self.stop_dist = rospy.get_param("~stop_distance", 0.25)
        self.slow_dist = rospy.get_param("~slow_distance", 0.5)
        self.ttc_stop = rospy.get_param("~ttc_stop", 0.6)
        self.ttc_slow = rospy.get_param("~ttc_slow", 1.2)
        self.front_arc_deg = rospy.get_param("~front_arc_deg", 30.0)
        self.cmd_sub = rospy.Subscriber("cmd_in", Twist, self._cmd_cb, queue_size=10)
        self.scan_sub = rospy.Subscriber("scan", LaserScan, self._scan_cb, queue_size=1)
        self.pub = rospy.Publisher("cmd_out", Twist, queue_size=10)
        self.last_scan = None
        self.last_cmd = Twist()

    def _scan_cb(self, msg: LaserScan):
        self.last_scan = msg

    def _cmd_cb(self, msg: Twist):
        self.last_cmd = msg
        safe_cmd = self._apply_safety(msg)
        self.pub.publish(safe_cmd)

    def _apply_safety(self, cmd: Twist) -> Twist:
        if self.last_scan is None:
            return cmd
        scan = self.last_scan
        angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
        arc = math.radians(self.front_arc_deg)
        mask = np.abs(angles) <= arc
        ranges = np.array(scan.ranges, dtype=float)
        ranges[~np.isfinite(ranges)] = scan.range_max
        ranges = np.clip(ranges, scan.range_min, scan.range_max)
        front = ranges[mask] if np.any(mask) else ranges
        d_min = float(np.min(front)) if front.size > 0 else float("inf")

        speed = max(0.0, cmd.linear.x)
        ttc = float("inf")
        if speed > 1e-3:
            closing = speed * np.maximum(0.0, np.cos(angles[mask])) if np.any(mask) else speed
            if np.any(closing > 1e-3):
                ttc_vals = ranges[mask] / closing if np.any(mask) else ranges / speed
                ttc = float(np.min(ttc_vals))

        out = Twist()
        out.linear = cmd.linear
        out.angular = cmd.angular

        # Distance-based
        if d_min <= self.stop_dist:
            out.linear.x = 0.0
            out.linear.y = 0.0
            out.angular.z = 0.0
            return out
        if d_min <= self.slow_dist:
            scale = max(0.2, (d_min - self.stop_dist) / max(1e-3, self.slow_dist - self.stop_dist))
            out.linear.x *= scale
            out.linear.y *= scale

        # TTC-based
        if math.isfinite(ttc):
            if ttc <= self.ttc_stop:
                out.linear.x = 0.0
                out.linear.y = 0.0
                out.angular.z = 0.0
            elif ttc <= self.ttc_slow:
                ratio = (ttc - self.ttc_stop) / max(1e-3, self.ttc_slow - self.ttc_stop)
                scale = max(0.2, ratio)
                out.linear.x *= scale
                out.linear.y *= scale
        return out


def main():
    rospy.init_node("cmd_vel_safety_filter")
    CmdVelSafety()
    rospy.loginfo("cmd_vel_safety_filter started")
    rospy.spin()


if __name__ == "__main__":
    main()
