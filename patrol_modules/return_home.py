#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
return_home.py ── 根據 odom / pose() 導回起點

用途
----
‣ 簡易 P 控制：僅依 odom 位置向量 → heading 誤差 → 線速 / 角速  
‣ 若前方再遇障礙（fused_front_distance < obs_thresh）→ 回傳 blocked=True
   讓主控切換 backoff，再回到返航。

使用
----
rh = ReturnHome(sensor_hub, cmd_pub, start_pose)
done, blocked = rh.step()       # 在主迴圈呼叫
"""
import math
import rospy
from geometry_msgs.msg import Twist
from .sensors import SensorHub

__all__ = ["ReturnHome"]


class ReturnHome:
    def __init__(
        self,
        sensor: SensorHub,
        cmd_pub,
        home_pose: tuple,
        max_lin: float = 0.25,
        max_ang: float = 0.6,
        goal_tol: float = 0.20,
        slow_dist: float = 0.60,
        obs_thresh: float = 0.50,
    ):
        """
        home_pose: (x0, y0) in odom/map frame
        """
        self.sensor = sensor
        self.pub = cmd_pub
        self.x0, self.y0 = home_pose

        self.v_max = max_lin
        self.w_max = max_ang
        self.goal_tol = goal_tol
        self.slow_dist = slow_dist
        self.obs_thresh = obs_thresh

        self.done = False

    # --------------------------------------------------
    def _publish(self, v, w):
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.pub.publish(twist)

    # --------------------------------------------------
    def step(self):
        """
        回傳 (done, blocked)
        • done=True  表示已到家
        • blocked=True 表示前方太近，需要 backoff
        """
        if self.done:
            return True, False

        # 1) 前方安全檢查
        front = self.sensor.fused_front_distance()
        if front is not None and front < self.obs_thresh:
            self._publish(0.0, 0.0)
            return False, True  # 需要 backoff

        # 2) 位置誤差
        x, y, yaw = self.sensor.pose()
        if x is None:
            self._publish(0.0, 0.0)
            return False, False  # 尚未取得 odom

        dx = self.x0 - x
        dy = self.y0 - y
        dist = math.hypot(dx, dy)

        if dist < self.goal_tol:
            self._publish(0.0, 0.0)
            self.done = True
            rospy.loginfo("ReturnHome: reached home (%.2fm)", dist)
            return True, False

        # 3) Heading 誤差
        tgt_yaw = math.atan2(dy, dx)
        yaw_err = math.atan2(math.sin(tgt_yaw - yaw), math.cos(tgt_yaw - yaw))

        # 4) 控制輸出
        # 角速：簡易 P (kp=1.0) + 飽和
        w = max(-self.w_max, min(self.w_max, yaw_err))

        # 線速：距離越近越慢，角度大亦降低
        v = self.v_max
        if dist < self.slow_dist:
            v = self.v_max * (dist / self.slow_dist)
        if abs(yaw_err) > math.radians(45):
            v *= 0.5

        self._publish(v, w)
        return False, False

