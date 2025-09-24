"""LiDAR based avoidance core with dynamic obstacle handling."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


@dataclass
class AvoidParams:
    holonomic: bool
    v_max: float
    v_min: float
    w_max: float
    ax_max: float
    aw_max: float
    base_radius: float
    safety_margin: float
    stop_distance: float
    slowdown_distance: float
    door_min_width: float
    range_min_valid: float
    range_max_valid: float
    temporal_median: int
    spatial_window: int
    method: str                  # "follow_the_gap" | "repulsive_field"
    goal_bias: float
    center_bias: float
    gap_min_length: float
    lookahead_distance: float
    yaw_kp: float
    curvature_gain: float
    clearance_gain: float
    min_clearance_keep: float
    # 動態障礙整合
    ttc_stop: float
    ttc_slow: float      # 秒
    dyn_inflation_base: float             # m（動態障礙基礎膨脹）
    dyn_inflation_gain: float             # m/(m/s) 速度越快膨脹越大
    proxemics_min: float                  # m 人際距離下限
    # 卡滯/恢復
    stuck_vel_eps: float
    stuck_time: float
    recovery_back: float
    recovery_spin: float
    recovery_trials: int
    publish_debug: bool


@dataclass
class GapInfo:
    theta_center: float
    theta_best: float
    width: float
    d_min: float


@dataclass
class DebugInfo:
    state: str
    d_min: float
    chosen_theta: float
    chosen_gap: GapInfo
    v_cmd: float
    w_cmd: float
    ttc_min: float
    stuck_flag: bool
    notes: str


class LidarAvoider:
    """Follow-the-gap avoider augmented with TTC and dynamic obstacles."""

    def __init__(self, params: AvoidParams):
        self.params = params
        self._scan_buffer: deque[np.ndarray] = deque(maxlen=max(1, params.temporal_median))
        self._filtered_ranges: Optional[np.ndarray] = None
        self._angle_min: float = -math.pi
        self._angle_increment: float = 0.0
        self._last_scan_stamp: float = 0.0
        self._odom: Optional[Odometry] = None
        self._nav_hint: Optional[Vector3] = None
        self._dyn_barriers: List[Dict] = []
        self._external_stop = False
        self._paused = False
        self._state = "IDLE"
        self._last_cmd = Twist()
        self._last_time = None
        self._stuck_start = None
        self._stuck_triggered = False
        self._stuck_reported = False
        self._recovery_phase = None
        self._recovery_end = 0.0
        self._recovery_trial = 0
        self._recovery_dir = 1

    # ------------------------------------------------------------------
    def update_scan(self, scan: LaserScan) -> None:
        ranges = np.array(scan.ranges, dtype=float)
        if ranges.size == 0:
            return
        ranges = np.clip(ranges, self.params.range_min_valid, self.params.range_max_valid)
        self._scan_buffer.append(ranges)

        filtered = self._temporal_filter()
        filtered = self._spatial_filter(filtered)
        self._filtered_ranges = filtered
        self._angle_min = scan.angle_min
        self._angle_increment = scan.angle_increment
        if scan.header.stamp and scan.header.stamp.to_sec() > 0.0:
            self._last_scan_stamp = scan.header.stamp.to_sec()
        else:
            self._last_scan_stamp = rospy.get_time()

    # ------------------------------------------------------------------
    def update_odom(self, odom: Odometry) -> None:
        self._odom = odom

    # ------------------------------------------------------------------
    def update_nav_hint(self, hint: Optional[Vector3]) -> None:
        self._nav_hint = hint

    # ------------------------------------------------------------------
    def ingest_dynamic_barriers(self, dyn_barriers: List[Dict]) -> None:
        self._dyn_barriers = dyn_barriers

    # ------------------------------------------------------------------
    def set_external_emergency_stop(self, flag: bool) -> None:
        self._external_stop = flag

    # ------------------------------------------------------------------
    def compute_cmd(self) -> Tuple[Twist, DebugInfo]:
        now = rospy.get_time()
        cmd = Twist()
        default_gap = GapInfo(theta_center=0.0, theta_best=0.0, width=0.0, d_min=self.params.stop_distance)
        debug = DebugInfo(
            state=self._state,
            d_min=float("inf"),
            chosen_theta=0.0,
            chosen_gap=default_gap,
            v_cmd=0.0,
            w_cmd=0.0,
            ttc_min=float("inf"),
            stuck_flag=self._stuck_triggered or self._stuck_reported,
            notes="",
        )

        if self._external_stop or self._paused:
            self._state = "PAUSE"
            debug.state = self._state
            debug.notes = "external_stop" if self._external_stop else "paused"
            self._reset_stuck_monitor()
            return cmd, debug

        if self._filtered_ranges is None:
            debug.notes = "waiting_scan"
            return cmd, debug

        effective_ranges = self._apply_dynamic_inflation(self._filtered_ranges.copy())
        inflation = self.params.base_radius + self.params.safety_margin
        clearance = np.maximum(0.0, effective_ranges - inflation)
        d_min = float(np.min(effective_ranges)) if effective_ranges.size else float("inf")
        debug.d_min = d_min

        ttc_min = self._estimate_ttc(effective_ranges)
        debug.ttc_min = ttc_min

        if self._state == "RECOVERY":
            v_cmd, w_cmd, notes = self._recovery_command(now)
            cmd.linear.x = v_cmd
            cmd.angular.z = w_cmd
            debug.state = self._state
            debug.notes = notes
            debug.v_cmd = v_cmd
            debug.w_cmd = w_cmd
            debug.stuck_flag = self._stuck_reported
            self._apply_rate_limits(cmd, now)
            return cmd, debug

        gaps = self._find_gaps(clearance, effective_ranges)
        chosen_theta = 0.0
        chosen_gap = default_gap
        notes = ""

        if gaps:
            chosen_gap, chosen_theta = self._select_gap(gaps)
            debug.chosen_gap = chosen_gap
            debug.chosen_theta = chosen_theta
        else:
            notes = "no_gap"
            chosen_gap = GapInfo(theta_center=0.0, theta_best=0.0, width=0.0, d_min=d_min)
            debug.chosen_gap = chosen_gap

        state = "CRUISE"
        v_target = self.params.v_max
        slowdown_factor = 1.0

        if d_min <= max(self.params.stop_distance, inflation) or ttc_min <= self.params.ttc_stop:
            state = "STOP"
            v_target = 0.0
            slowdown_factor = 0.0
            notes = notes or "hard_stop"
        else:
            dist_factor = 1.0
            if d_min < self.params.slowdown_distance:
                dist_factor = max(
                    0.0,
                    (d_min - self.params.stop_distance)
                    / max(1e-3, self.params.slowdown_distance - self.params.stop_distance),
                )
            ttc_factor = 1.0
            if ttc_min < self.params.ttc_slow:
                ttc_factor = max(
                    0.0,
                    (ttc_min - self.params.ttc_stop)
                    / max(1e-3, self.params.ttc_slow - self.params.ttc_stop),
                )
            slowdown_factor = min(dist_factor, ttc_factor)
            if slowdown_factor < 1.0:
                state = "SLOW"
            v_target = self.params.v_max * slowdown_factor

        if chosen_gap.width <= 0.0:
            chosen_theta = 0.0
        v_cmd = v_target
        if v_cmd > 0.0:
            v_cmd = max(self.params.v_min, min(self.params.v_max, v_cmd))
            # Curve slowdown and clearance bias
            v_cmd /= 1.0 + self.params.curvature_gain * abs(chosen_theta)
            clearance_bonus = 0.0
            if math.isfinite(chosen_gap.d_min):
                clearance_bonus = max(0.0, chosen_gap.d_min - self.params.stop_distance)
            v_cmd += self.params.clearance_gain * clearance_bonus
            v_cmd = max(self.params.v_min, min(self.params.v_max, v_cmd))
            if chosen_gap.d_min < self.params.min_clearance_keep:
                v_cmd = min(v_cmd, self.params.v_min)
        else:
            v_cmd = 0.0

        w_cmd = self.params.yaw_kp * chosen_theta
        w_cmd = max(-self.params.w_max, min(self.params.w_max, w_cmd))

        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd

        debug.state = state
        debug.v_cmd = v_cmd
        debug.w_cmd = w_cmd
        debug.notes = notes

        self._monitor_stuck(now, v_cmd)
        debug.stuck_flag = self._stuck_triggered or self._stuck_reported

        self._apply_rate_limits(cmd, now)
        self._state = state

        return cmd, debug

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._scan_buffer.clear()
        self._filtered_ranges = None
        self._nav_hint = None
        self._dyn_barriers = []
        self._external_stop = False
        self._paused = False
        self._state = "IDLE"
        self._last_cmd = Twist()
        self._last_time = None
        self._reset_stuck_monitor(full=True)

    # ------------------------------------------------------------------
    def pause(self, enabled: bool) -> None:
        self._paused = enabled
        if enabled:
            self._state = "PAUSE"
            self._reset_stuck_monitor()

    # ------------------------------------------------------------------
    def is_stuck(self) -> bool:
        return self._stuck_reported

    # ------------------------------------------------------------------
    def current_state(self) -> str:
        return self._state

    # ------------------------------------------------------------------
    def _temporal_filter(self) -> np.ndarray:
        window = max(1, self.params.temporal_median)
        if window <= 1 or len(self._scan_buffer) == 1:
            return self._scan_buffer[-1]
        stack = list(self._scan_buffer)[-window:]
        return np.median(np.stack(stack, axis=0), axis=0)

    # ------------------------------------------------------------------
    def _spatial_filter(self, ranges: np.ndarray) -> np.ndarray:
        window = max(1, self.params.spatial_window)
        if window <= 1:
            return ranges
        radius = window // 2
        padded = np.pad(ranges, (radius, radius), mode="edge")
        filtered = np.empty_like(ranges)
        for i in range(len(ranges)):
            segment = padded[i : i + window]
            filtered[i] = np.median(segment)
        return filtered

    # ------------------------------------------------------------------
    def _apply_dynamic_inflation(self, ranges: np.ndarray) -> np.ndarray:
        for barrier in self._dyn_barriers:
            rng = float(barrier.get("range", 0.0))
            theta = float(barrier.get("theta", 0.0))
            radius = float(barrier.get("radius", 0.0))
            if rng <= 1e-3:
                continue
            if radius < self.params.proxemics_min:
                radius = self.params.proxemics_min
            idx = int(round((theta - self._angle_min) / max(1e-6, self._angle_increment)))
            spread = int(math.ceil(radius / max(1e-3, rng * self._angle_increment)))
            for offset in range(-spread, spread + 1):
                ii = idx + offset
                if 0 <= ii < len(ranges):
                    ranges[ii] = min(ranges[ii], max(self.params.range_min_valid, rng - radius))
        return ranges

    # ------------------------------------------------------------------
    def _find_gaps(self, clearance: np.ndarray, effective_ranges: np.ndarray) -> List[GapInfo]:
        gaps: List[GapInfo] = []
        i = 0
        n = len(clearance)
        while i < n:
            if clearance[i] <= 0.0:
                i += 1
                continue
            start = i
            best_idx = i
            best_val = clearance[i]
            while i + 1 < n and clearance[i + 1] > 0.0:
                i += 1
                if clearance[i] > best_val:
                    best_val = clearance[i]
                    best_idx = i
            end = i
            width = self._estimate_gap_width(start, end, effective_ranges)
            theta_center = self._angle_min + ((start + end) / 2.0) * self._angle_increment
            theta_best = self._angle_min + best_idx * self._angle_increment
            d_min = float(np.min(effective_ranges[start : end + 1]))
            if width >= self.params.gap_min_length and width >= self.params.door_min_width:
                if d_min >= self.params.min_clearance_keep:
                    gaps.append(GapInfo(theta_center=theta_center, theta_best=theta_best, width=width, d_min=d_min))
            i += 1
        return gaps

    # ------------------------------------------------------------------
    def _estimate_gap_width(self, start: int, end: int, ranges: np.ndarray) -> float:
        if end <= start:
            return 0.0
        segment = ranges[start : end + 1]
        mean_range = float(np.mean(segment))
        span = (end - start + 1) * self._angle_increment
        return abs(mean_range * span)

    # ------------------------------------------------------------------
    def _select_gap(self, gaps: List[GapInfo]) -> Tuple[GapInfo, float]:
        hint_theta = self._nav_hint_theta()
        best_score = -float("inf")
        best_gap = gaps[0]
        best_theta = 0.0
        for gap in gaps:
            theta_gap = self._blend_angles(gap.theta_best, gap.theta_center, self.params.center_bias)
            theta_ref = theta_gap
            align_score = 0.0
            if hint_theta is not None:
                theta_ref = self._blend_angles(theta_gap, hint_theta, self.params.goal_bias)
                align_score = math.cos(theta_ref - hint_theta)
            clearance_score = gap.width + gap.d_min
            score = clearance_score + align_score
            if score > best_score:
                best_score = score
                best_gap = gap
                best_theta = theta_ref
        return best_gap, best_theta

    # ------------------------------------------------------------------
    def _nav_hint_theta(self) -> Optional[float]:
        if self._nav_hint is None:
            return None
        mag = math.hypot(self._nav_hint.x, self._nav_hint.y)
        if mag < 1e-3:
            return None
        return math.atan2(self._nav_hint.y, self._nav_hint.x)

    # ------------------------------------------------------------------
    def _blend_angles(self, a: float, b: float, weight: float) -> float:
        weight = max(0.0, min(1.0, weight))
        diff = self._angle_wrap(b - a)
        return self._angle_wrap(a + weight * diff)

    # ------------------------------------------------------------------
    @staticmethod
    def _angle_wrap(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    # ------------------------------------------------------------------
    def _estimate_ttc(self, effective_ranges: np.ndarray) -> float:
        ttc_min = float("inf")
        robot_speed = 0.0
        if self._odom is not None:
            robot_speed = self._odom.twist.twist.linear.x
        forward_mask = self._forward_indices(len(effective_ranges))
        for idx in forward_mask:
            rng = effective_ranges[idx]
            if rng <= 0.0:
                continue
            if robot_speed <= 1e-3:
                continue
            ttc = rng / max(robot_speed, 1e-3)
            if ttc < ttc_min:
                ttc_min = ttc
        for barrier in self._dyn_barriers:
            rng = max(0.0, float(barrier.get("range", 0.0)) - float(barrier.get("radius", 0.0)))
            v_rel = float(barrier.get("v_rel", 0.0))
            if rng <= 0.0 or v_rel <= 1e-3:
                continue
            ttc = rng / v_rel
            if ttc < ttc_min:
                ttc_min = ttc
        return ttc_min

    # ------------------------------------------------------------------
    def _forward_indices(self, n: int) -> List[int]:
        if self._angle_increment == 0.0:
            return list(range(n))
        forward = []
        for i in range(n):
            theta = self._angle_min + i * self._angle_increment
            if abs(theta) <= math.pi / 2:
                forward.append(i)
        return forward

    # ------------------------------------------------------------------
    def _monitor_stuck(self, now: float, v_cmd: float) -> None:
        speed = 0.0
        if self._odom is not None:
            vx = self._odom.twist.twist.linear.x
            vy = self._odom.twist.twist.linear.y
            speed = math.hypot(vx, vy)
        if v_cmd > self.params.stuck_vel_eps and speed < self.params.stuck_vel_eps:
            if self._stuck_start is None:
                self._stuck_start = now
            elif now - self._stuck_start > self.params.stuck_time and not self._stuck_triggered:
                self._start_recovery(now)
        else:
            self._stuck_start = None
            if self._state != "RECOVERY":
                self._stuck_triggered = False

    # ------------------------------------------------------------------
    def _start_recovery(self, now: float) -> None:
        self._stuck_triggered = True
        self._state = "RECOVERY"
        self._recovery_phase = "BACK"
        self._recovery_trial += 1
        self._recovery_dir *= -1
        back_speed = max(self.params.v_min, 0.05)
        duration = self.params.recovery_back / max(back_speed, 1e-3)
        self._recovery_end = now + duration
        self._stuck_start = None
        if self._recovery_trial > self.params.recovery_trials:
            self._stuck_reported = True

    # ------------------------------------------------------------------
    def _recovery_command(self, now: float) -> Tuple[float, float, str]:
        if self._stuck_reported:
            return 0.0, 0.0, "recovery_failed"
        if self._recovery_phase == "BACK":
            v = -min(self.params.v_min, self.params.v_max)
            if now >= self._recovery_end:
                self._recovery_phase = "SPIN"
                spin_rate = 0.6 * self.params.w_max
                duration = abs(self.params.recovery_spin) / max(spin_rate, 1e-3)
                self._recovery_end = now + duration
            return v, 0.0, "recovery_back"
        if self._recovery_phase == "SPIN":
            spin_rate = 0.6 * self.params.w_max
            w = self._recovery_dir * max(0.2, spin_rate)
            if now >= self._recovery_end:
                self._state = "CRUISE"
                self._recovery_phase = None
                self._stuck_triggered = False
                self._recovery_trial = 0
                return 0.0, 0.0, "recovery_reset"
            return 0.0, w, "recovery_spin"
        self._state = "CRUISE"
        return 0.0, 0.0, "recovery_idle"

    # ------------------------------------------------------------------
    def _apply_rate_limits(self, cmd: Twist, now: float) -> None:
        if self._last_time is None:
            self._last_time = now
            last = Twist()
            last.linear.x = cmd.linear.x
            last.angular.z = cmd.angular.z
            self._last_cmd = last
            return
        dt = max(1e-3, now - self._last_time)
        dv = self.params.ax_max * dt
        dw = self.params.aw_max * dt
        cmd.linear.x = self._clip(cmd.linear.x, self._last_cmd.linear.x - dv, self._last_cmd.linear.x + dv)
        cmd.angular.z = self._clip(cmd.angular.z, self._last_cmd.angular.z - dw, self._last_cmd.angular.z + dw)
        last = Twist()
        last.linear.x = cmd.linear.x
        last.angular.z = cmd.angular.z
        self._last_cmd = last
        self._last_time = now

    # ------------------------------------------------------------------
    @staticmethod
    def _clip(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    # ------------------------------------------------------------------
    def _reset_stuck_monitor(self, full: bool = False) -> None:
        self._stuck_start = None
        self._stuck_triggered = False
        if full:
            self._stuck_reported = False
            self._recovery_trial = 0
            self._recovery_phase = None
            self._recovery_dir = 1

