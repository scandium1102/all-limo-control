#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backoff.py ── 遇前方障礙時的「倒退 → 判斷 → 轉向」行為  
**v3‑odom (2025‑07‑09)**
--------------------------------------------------
1. **turn_dir = "auto"**：倒退到安全距離後，自動比較左右空間決定轉向。
2. 新增 **decide** 階段：透過 LiDAR / 融合距離評估左、右最小距離。
3. Phase 流程：`idle → back → decide → turn → done`。
4. ✨ **加入 odom/IMU 迴授**：`turn` 階段精準轉到目標角度（預設 85°），缺訊則退回 timeout。
5. 參數化 `odom_topic`、`target_angle_deg`，其餘介面不變。
"""

import math
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

from .sensors import SensorHub

__all__ = ["BackoffBehavior"]


class BackoffBehavior:
    """倒退至安全距離，再依左右空間自動（或指定）轉向離開障礙

    僅依賴 odom/IMU 角度回授即可精準轉向；若 odom 不可用，仍可
    退回計時開迴路邏輯（`turn_timeout`）。
    """

    def __init__(
        self,
        sensor: SensorHub,
        cmd_pub,
        *,
        safe_dist: float = 0.90,
        back_speed: float = 0.12,
        turn_speed: float = 0.60,
        turn_linear: float = 0.08,
        back_timeout: float = 4.0,
        decide_pause: float = 0.25,
        turn_timeout: float = 6.0,
        odom_topic: str = "/odom",
        target_angle_deg: float = 100.0,
    ):
        # 依賴物件 & topic
        self.sensor = sensor
        self.pub = cmd_pub
        self._odom_sub = rospy.Subscriber(odom_topic, Odometry, self._odom_cb, queue_size=10)

        # 參數
        self.safe = safe_dist
        self.v_back = -abs(back_speed)
        self.v_turn = abs(turn_linear)
        self.w_turn = abs(turn_speed)
        self.back_timeout = back_timeout
        self.decide_pause = decide_pause
        self.turn_timeout = turn_timeout
        self.target_angle = math.radians(abs(target_angle_deg))  # 目標轉角 (rad)

        # 狀態
        self._phase = "idle"      # idle | back | decide | turn | done
        self._start_ts = None
        self._dir = "left"        # left | right | auto
        self.done = False
        self.result = "none"

        # odom
        self._yaw = None           # 實時偏航角 (rad)
        self._yaw0 = None          # 進入 turn 當下的偏航角

    # --------------------------------------------------
    # ROS Callbacks
    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._yaw = yaw

    # --------------------------------------------------
    # Public API
    def start(self, turn_dir: str = "auto"):
        """初始化並啟動 backoff。

        :param turn_dir: "left" / "right" / "auto"（預設）。
        """
        self._phase = "back"
        self._start_ts = rospy.Time.now()
        self._dir = turn_dir
        self.done = False
        self.result = "running"
        self._yaw0 = None

    def cancel(self):
        """外部強制中止"""
        self._publish(0.0, 0.0)
        self.done = True
        self.result = "cancelled"

    # --------------------------------------------------
    # Internal helpers
    def _publish(self, v: float, w: float):
        tw = Twist()
        tw.linear.x = v
        tw.angular.z = w
        self.pub.publish(tw)

    def _choose_turn_dir(self) -> str:
        """比較左右空間，回傳 'left' 或 'right'。"""
        left = self.sensor.arc_min_distance(45, 110)
        right = self.sensor.arc_min_distance(-110, -45)
        left = left if left is not None else 999.0
        right = right if right is not None else 999.0
        return "left" if left > right else "right"

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """回傳 a‑b 的最小絕對角差 (0~π)"""
        return abs(math.atan2(math.sin(a - b), math.cos(a - b)))

    # --------------------------------------------------
    # Main step
    def step(self):
        """每次迴圈呼叫；自動推進流程並直接 publish cmd_vel"""
        if self.done or self._phase == "idle":
            return

        now = rospy.Time.now()
        dt = (now - self._start_ts).to_sec()

        # ---------- Phase 1: back ----------
        if self._phase == "back":
            self._publish(self.v_back, 0.0)
            front = self.sensor.fused_front_distance()
            reached_safe = front and front > self.safe
            if reached_safe or dt >= self.back_timeout:
                self._phase = "decide"
                self._start_ts = now
                self._publish(0.0, 0.0)  # stop & settle
                return

        # ---------- Phase 2: decide ----------
        elif self._phase == "decide":
            if dt < self.decide_pause:
                return  # wait for settling
            if self._dir == "auto":
                self._dir = self._choose_turn_dir()
            self._phase = "turn"
            self._start_ts = now
            self._yaw0 = self._yaw  # may be None; handle later
            return

        # ---------- Phase 3: turn (odom‑guided) ----------
        elif self._phase == "turn":
            w = self.w_turn if self._dir == "left" else -self.w_turn
            self._publish(self.v_turn, w)

            # 判斷是否達成目標角度
            done_by_angle = False
            if self._yaw is not None and self._yaw0 is not None:
                yaw_diff = self._angle_diff(self._yaw, self._yaw0)
                done_by_angle = yaw_diff >= self.target_angle

            done_by_time = dt >= self.turn_timeout

            if done_by_angle or done_by_time:
                self._publish(0.0, 0.0)
                self._phase = "done"
                self.done = True
                self.result = "ok"

