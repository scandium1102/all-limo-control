#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stuck_recovery.py ── 偵測卡死並執行脫困

偵測邏輯
--------
連續 `check_window` 秒內，里程計位移 < `move_thresh` (m) → 視為卡死

脫困策略
--------
1. 後退 `back_time` 秒  (速度 -back_speed)
2. 隨機左右旋轉 `spin_angle` (90°–180°)：
     - 以 `spin_speed` 角速度
     - 同時帶輕微線速 `spin_linear`
3. 完成後設 done=True，主控可恢復正常行為

用法
----
recovery = StuckRecovery(sensor_hub, cmd_pub)
recovery.monitor()            # 每迴圈呼叫，回傳 (stuck_detected, done)
if stuck_detected:
    recovery.execute()        # 在 while 迴圈中呼叫直到 done
"""

import math
import random
import rospy
from geometry_msgs.msg import Twist
from .sensors import SensorHub

__all__ = ["StuckRecovery"]


class StuckRecovery:
    def __init__(
        self,
        sensor: SensorHub,
        cmd_pub,
        move_thresh: float = 0.05,  # m
        check_window: float = 2.0,  # s
        back_speed: float = 0.10,
        back_time: float = 1.0,
        spin_speed: float = 0.6,
        spin_linear: float = 0.15,
    ):
        self.sensor = sensor
        self.pub = cmd_pub

        self.move_thresh = move_thresh
        self.check_window = check_window
        self.back_speed = -abs(back_speed)
        self.back_time = back_time
        self.spin_speed = abs(spin_speed)
        self.spin_linear = spin_linear

        # 狀態
        self._last_pos = None
        self._last_ts = rospy.Time.now()
        self._in_recovery = False
        self._phase = "idle"
        self._phase_ts = None
        self._spin_duration = None
        self.done = False

    # --------------------------------------------------
    def _publish(self, v, w):
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.pub.publish(twist)

    # --------------------------------------------------
    def monitor(self):
        """
        每次主迴圈呼叫：
        回傳 (stuck_detected, recovery_done)
        """
        if self._in_recovery:
            # 執行中
            return True, False

        now = rospy.Time.now()
        x, y, _ = self.sensor.pose()
        if x is None:
            return False, False  # 無 odom

        if self._last_pos is None:
            self._last_pos = (x, y)
            self._last_ts = now
            return False, False

        dist = math.hypot(x - self._last_pos[0], y - self._last_pos[1])
        if dist >= self.move_thresh:
            # 有移動 → 重設 baseline
            self._last_pos = (x, y)
            self._last_ts = now
            return False, False

        if (now - self._last_ts).to_sec() >= self.check_window:
            # 判定卡死
            self._in_recovery = True
            self._phase = "back"
            self._phase_ts = now
            self.done = False
            rospy.logwarn("Stuck detected → start recovery")
            return True, False

        return False, False

    # --------------------------------------------------
    def execute(self):
        """
        在主迴圈中持續呼叫，直到 self.done=True。
        monitor() 應先判定卡死才呼叫本函式。
        """
        if not self._in_recovery:
            return

        now = rospy.Time.now()

        # ---------- Phase 1: back ----------
        if self._phase == "back":
            self._publish(self.back_speed, 0.0)
            if (now - self._phase_ts).to_sec() >= self.back_time:
                # 計算隨機旋轉時長
                angle = random.uniform(math.pi / 2, math.pi)
                self._spin_duration = angle / self.spin_speed
                self._phase = "spin"
                self._phase_ts = now
                self._spin_dir = random.choice(["left", "right"])
                rospy.loginfo("Recovery spin %.0f° %s", math.degrees(angle), self._spin_dir)

        # ---------- Phase 2: spin ----------
        elif self._phase == "spin":
            w = self.spin_speed if self._spin_dir == "left" else -self.spin_speed
            self._publish(self.spin_linear, w)
            if (now - self._phase_ts).to_sec() >= self._spin_duration:
                self._publish(0.0, 0.0)
                self._phase = "done"
                self.done = True
                self._in_recovery = False
                # 重設 baseline
                self._last_pos = self.sensor.pose()[:2]
                self._last_ts = now
                rospy.loginfo("Recovery done")


