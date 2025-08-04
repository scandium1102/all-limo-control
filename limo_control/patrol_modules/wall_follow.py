#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wall_follow.py ── 右手貼牆行為

使用方式︰
    hub = SensorHub(...)
    wf  = WallFollower(hub)
    twist, blocked = wf.compute_cmd()
    if blocked:
        # 交給 backoff 行為
    else:
        cmd_pub.publish(twist)

特色︰
1. P‒I‒D 控制角速度，維持右側距離 target_dist。
2. 自動根據前方距離調整線速度（越靠近障礙越慢）。
3. 當前方 fused_front_distance < obs_thresh 時回傳 blocked=True，
   讓主控切換到避障行為。
"""

import math
import rospy
from geometry_msgs.msg import Twist
from .sensors import SensorHub
from .coverage_grid import PID  # 直接重複利用已定義 PID；若無請 from patrol_modules.sensors import PID

__all__ = ["WallFollower"]


class WallFollower:
    """右手貼牆控制器"""

    def __init__(
        self,
        sensor: SensorHub,
        target_dist: float = 0.45,
        fwd_speed: float = 0.25,
        slow_dist: float = 1.0,
        obs_thresh: float = 0.50,
        kp: float = 1.2,
        ki: float = 0.0,
        kd: float = 0.2,
        max_yaw: float = 0.6,
    ):
        self.sensor = sensor
        self.target = target_dist
        self.v_max = fwd_speed
        self.slow_dist = slow_dist
        self.obs_thresh = obs_thresh
        self.pid = PID(kp, ki, kd, limit=max_yaw)
        self._rate_hz = rospy.get_param("~rate_hz", 20)

    # --------------------------------------------------
    def compute_cmd(self):
        """
        回傳 (Twist, blocked)
        blocked=True 表示前方障礙需切換至 backoff 行為
        """
        cmd = Twist()

        # ---------- 1) 前方障礙判斷 ----------
        front = self.sensor.fused_front_distance()
        if front is not None and front < self.obs_thresh:
            return cmd, True  # 交給 backoff

        # ---------- 2) 線速度調整 ----------
        v = self.v_max
        if front is not None and front < self.slow_dist:
            if front <= self.obs_thresh:
                v = 0.0
            else:
                frac = (front - self.obs_thresh) / (
                    self.slow_dist - self.obs_thresh
                )
                v = max(0.05, frac * self.v_max)
        cmd.linear.x = v

        # ---------- 3) 角速度 PID ----------
        rd = self.sensor.right_distance()
        if rd is None:
            # 找不到牆 → 讓機器人稍微向右探索牆
            err = -0.2  # 假設太遠，向右轉
        else:
            err = self.target - rd
        yaw_rate = self.pid.step(err, 1.0 / self._rate_hz)
        cmd.angular.z = yaw_rate

        return cmd, False

