#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensors.py ── LIMO SensorHub

封裝：
    • 2D LiDAR    : /limo/scan           (sensor_msgs/LaserScan)
    • 深度點雲    : /camera/depth/points (sensor_msgs/PointCloud2)
    • 里程計(Odom): /odom                (nav_msgs/Odometry)

對外提供：
    SensorHub.ready              -> bool   監聽是否收到第一筆資料
    SensorHub.pose()             -> (x, y, yaw)
    SensorHub.front_distance()   -> float | None
    SensorHub.right_distance()   -> float | None
    SensorHub.fused_front_distance() -> float | None   # LiDAR+Depth

參數（可在主節點 rosparam 設定）：
    ~lidar_arc_deg          (30)   前方扇形 ±° 用於 front_distance
    ~depth_y_window         (0.20) 垂直視窗 (m) 過濾點雲
    ~depth_h_arc_deg        (30)   水平視窗 ±° 用於點雲
    ~right_center_deg       (-90)  右側牆中心角度
    ~right_half_deg         (10)   右側計算半寬
"""

import math
import rospy
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs import point_cloud2                 # ← 改成這一行
from nav_msgs.msg import Odometry

__all__ = ["SensorHub"]


class SensorHub:
    """集中管理 LiDAR / Depth / Odom，並提供即時查詢介面"""

    def __init__(
        self,
        scan_topic="/limo/scan",
        cloud_topic="/camera/depth/points",
        odom_topic="/odom",
    ):
        # 讀參數
        self.lidar_arc = math.radians(
            rospy.get_param("~lidar_arc_deg", 30.0)
        )  # 前方扇形
        self.depth_y_win = rospy.get_param("~depth_y_window", 0.20)
        self.depth_h_arc = math.radians(
            rospy.get_param("~depth_h_arc_deg", 30.0)
        )
        self.right_center = math.radians(
            rospy.get_param("~right_center_deg", -90.0)
        )
        self.right_half = math.radians(
            rospy.get_param("~right_half_deg", 10.0)
        )

        # 資料暫存
        self.scan = None
        self.cloud = None
        self.x = self.y = self.yaw = None

        # ROS subscriber
        rospy.Subscriber(scan_topic, LaserScan, self._cb_scan, queue_size=1)
        rospy.Subscriber(cloud_topic, PointCloud2, self._cb_cloud, queue_size=1)
        rospy.Subscriber(odom_topic, Odometry, self._cb_odom, queue_size=10)

    # ---------- property ----------
    @property
    def ready(self):
        """是否已收到 scan 與 odom"""
        return self.scan is not None and self.x is not None

    # ---------- Callbacks ----------
    def _cb_scan(self, msg):
        self.scan = msg

    def _cb_cloud(self, msg):
        self.cloud = msg

    def _cb_odom(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        # quaternion -> yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    # ---------- Public API ----------
    def pose(self):
        """回傳 (x, y, yaw)。可能為 (None, None, None)。"""
        return self.x, self.y, self.yaw

    def front_distance(self):
        """僅使用 LiDAR：回傳前方 ±arc 度內最小距離。"""
        if not self.scan:
            return None

        angle = self.scan.angle_min
        min_r = float("inf")
        for r in self.scan.ranges:
            if -self.lidar_arc <= angle <= self.lidar_arc and r > 0.0:
                min_r = min(min_r, r)
            angle += self.scan.angle_increment

        return min_r if min_r != float("inf") else None

    def right_distance(self):
        """僅使用 LiDAR：回傳右側指定中心角 ±half 度內最小距離。"""
        if not self.scan:
            return None

        angle = self.scan.angle_min
        min_r = float("inf")
        for r in self.scan.ranges:
            if (
                self.right_center - self.right_half
                <= angle
                <= self.right_center + self.right_half
                and r > 0.0
            ):
                min_r = min(min_r, r)
            angle += self.scan.angle_increment

        return min_r if min_r != float("inf") else None

    # ---------- LiDAR + Depth 融合 ----------
    def fused_front_distance(self):
        """
        回傳前方最近障礙距離：
          • 先計算 LiDAR ±arc 最近值
          • 再掃描點雲 (|y|<depth_y_window, |x|<tan(h_arc)*z)
          • 取兩者最小
        """
        d = self.front_distance()
        d = float("inf") if d is None else d

        # 使用深度點雲補盲區
        if self.cloud:
            h_limit = math.tan(self.depth_h_arc)
            try:
                for x, y, z in point_cloud2.read_points(
                    self.cloud, field_names=("x", "y", "z"), skip_nans=True
                ):
                    if z <= 0:
                        continue  # 過濾攝像機後方點
                    if abs(y) > self.depth_y_win:
                        continue
                    if abs(x) > h_limit * z:
                        continue
                    dist = math.sqrt(x * x + y * y + z * z)
                    if dist < d:
                        d = dist
                        if d < 0.10:  # 足夠近，直接返回
                            break
            except Exception as e:
                rospy.logerr_throttle(5, "PointCloud parse error: %s", e)

        return None if d == float("inf") else d

