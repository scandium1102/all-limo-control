"""LiDAR based reactive avoidance controller for LIMO."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
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
    method: str
    goal_bias: float
    center_bias: float
    gap_min_length: float
    lookahead_distance: float
    yaw_kp: float
    curvature_gain: float
    clearance_gain: float
    min_clearance_keep: float
    ttc_stop: float
    ttc_slow: float
    dyn_inflation_base: float
    dyn_inflation_gain: float
    proxemics_min: float
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
    chosen_gap: Optional[GapInfo]
    v_cmd: float
    w_cmd: float
    ttc_min: float
    stuck_flag: bool
    notes: str


class LidarAvoider:
    """Follow-the-gap avoider with TTC and recovery logic."""

    def __init__(self, params: AvoidParams):
        self.params = params
        self._scan_history: deque[np.ndarray] = deque(maxlen=max(1, params.temporal_median))
        self._filtered_ranges: Optional[np.ndarray] = None
        self._angles: Optional[np.ndarray] = None
        self._scan_meta: Optional[Tuple[float, float, float]] = None
        self._nav_hint_theta: Optional[float] = None
        self._dyn_barriers: List[Dict] = []
        self._pause = False
        self._external_stop = False
        self._state = "IDLE"
        self._last_cmd = Twist()
        self._last_cmd_time = self._time()
        self._robot_speed = 0.0
        self._robot_pose = (0.0, 0.0, 0.0)
        self._stuck_start: Optional[float] = None
        self._stuck_flag = False
        self._recovery_trial = 0
        self._recovery_active = False
        self._recovery_phase = "back"
        self._recovery_end_time = 0.0
        self._recovery_spin_dir = 1
        self._last_gap: Optional[GapInfo] = None

    # ------------------------------------------------------------------
    def update_scan(self, scan: LaserScan) -> None:
        if not scan.ranges:
            return

        ranges = np.array(scan.ranges, dtype=float)
        finite_mask = np.isfinite(ranges)
        ranges[~finite_mask] = scan.range_max if scan.range_max > 0 else self.params.range_max_valid

        valid_min = max(scan.range_min, self.params.range_min_valid)
        valid_max = min(scan.range_max if scan.range_max > 0 else self.params.range_max_valid, self.params.range_max_valid)
        ranges = np.clip(ranges, valid_min, valid_max)

        self._scan_history.append(ranges)
        if len(self._scan_history) == 0:
            return

        stacked = np.stack(list(self._scan_history), axis=0)
        median_ranges = np.median(stacked, axis=0)

        if self.params.spatial_window > 1:
            half = self.params.spatial_window // 2
            padded = np.pad(median_ranges, (half,), mode="edge")
            smoothed = np.empty_like(median_ranges)
            window = self.params.spatial_window
            for i in range(len(median_ranges)):
                smoothed[i] = np.median(padded[i : i + window])
            ranges_proc = smoothed
        else:
            ranges_proc = median_ranges

        inflation = self.params.base_radius + self.params.safety_margin
        ranges_proc = np.maximum(0.0, ranges_proc - inflation)

        count = ranges_proc.shape[0]
        angles = scan.angle_min + np.arange(count, dtype=float) * scan.angle_increment

        self._filtered_ranges = ranges_proc
        self._angles = angles
        self._scan_meta = (scan.angle_min, scan.angle_increment, float(count))

    # ------------------------------------------------------------------
    def update_odom(self, odom: Odometry) -> None:
        twist = odom.twist.twist
        self._robot_speed = math.sqrt(twist.linear.x ** 2 + twist.linear.y ** 2)

        pose = odom.pose.pose
        yaw = self._quaternion_to_yaw(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        self._robot_pose = (pose.position.x, pose.position.y, yaw)

    # ------------------------------------------------------------------
    def update_nav_hint(self, hint: Optional[Vector3]) -> None:
        if hint is None:
            self._nav_hint_theta = None
            return
        mag = math.hypot(hint.x, hint.y)
        if mag < 1e-6:
            self._nav_hint_theta = None
        else:
            self._nav_hint_theta = math.atan2(hint.y, hint.x)

    # ------------------------------------------------------------------
    def ingest_dynamic_barriers(self, dyn_barriers: List[Dict]) -> None:
        self._dyn_barriers = list(dyn_barriers)

    # ------------------------------------------------------------------
    def set_external_emergency_stop(self, flag: bool) -> None:
        self._external_stop = bool(flag)

    # ------------------------------------------------------------------
    def compute_cmd(self) -> Tuple[Twist, DebugInfo]:
        now = self._time()
        cmd = Twist()
        debug = DebugInfo(
            state=self._state,
            d_min=float("inf"),
            chosen_theta=0.0,
            chosen_gap=None,
            v_cmd=0.0,
            w_cmd=0.0,
            ttc_min=float("inf"),
            stuck_flag=self._stuck_flag,
            notes="",
        )

        if self._external_stop:
            self._state = "EMERGENCY_STOP"
            debug.state = self._state
            debug.notes = "External emergency stop"
            return cmd, debug

        if self._pause:
            self._state = "PAUSED"
            debug.state = self._state
            debug.notes = "Paused"
            return cmd, debug

        if self._filtered_ranges is None or self._angles is None:
            debug.notes = "No scan"
            return cmd, debug

        ranges = self._filtered_ranges.copy()
        angles = self._angles

        self._apply_dynamic_inflation(ranges)
        d_min = float(np.min(ranges)) if ranges.size else float("inf")
        debug.d_min = d_min

        if not np.isfinite(d_min) or d_min <= 0.0:
            d_min = 0.0

        if self._recovery_active:
            cmd, note = self._do_recovery(now)
            debug.state = self._state
            debug.notes = note
            debug.v_cmd = cmd.linear.x
            debug.w_cmd = cmd.angular.z
            debug.chosen_gap = self._last_gap
            self._last_cmd = cmd
            self._last_cmd_time = now
            return cmd, debug

        gaps = self._find_gaps(ranges, angles)
        if not gaps:
            self._state = "STOP"
            debug.state = self._state
            debug.notes = "No navigable gap"
            self._check_stuck(now, 0.0)
            self._last_cmd = cmd
            self._last_cmd_time = now
            return cmd, debug

        gap = self._select_gap(gaps, angles)
        self._last_gap = gap
        theta_gap = (1.0 - self.params.center_bias) * gap.theta_best + self.params.center_bias * gap.theta_center
        theta_ref = theta_gap
        if self._nav_hint_theta is not None:
            theta_ref = (1.0 - self.params.goal_bias) * theta_ref + self.params.goal_bias * self._nav_hint_theta
        theta_ref = max(-math.pi, min(math.pi, theta_ref))

        w_cmd = self.params.yaw_kp * theta_ref
        w_cmd = max(-self.params.w_max, min(self.params.w_max, w_cmd))

        v_cmd = self._compute_speed(d_min, abs(theta_ref))

        ttc_min, v_cmd, ttc_note = self._apply_ttc(v_cmd, ranges, angles)

        debug.ttc_min = ttc_min
        if ttc_note:
            debug.notes = ttc_note

        dt = max(1e-3, now - self._last_cmd_time)
        v_cmd = self._limit_rate(v_cmd, self._last_cmd.linear.x, self.params.ax_max, dt)
        w_cmd = self._limit_rate(w_cmd, self._last_cmd.angular.z, self.params.aw_max, dt)

        cmd.linear.x = v_cmd
        if self.params.holonomic:
            cmd.linear.y = 0.0
        cmd.angular.z = w_cmd

        self._state = "CRUISE" if v_cmd > self.params.v_min else "SLOW" if v_cmd > 0.0 else "STOP"
        debug.state = self._state
        debug.chosen_theta = theta_ref
        debug.v_cmd = v_cmd
        debug.w_cmd = w_cmd
        debug.chosen_gap = gap

        self._check_stuck(now, v_cmd)
        debug.stuck_flag = self._stuck_flag
        if self._stuck_flag and not self._recovery_active:
            self._start_recovery(now)
            debug.notes = "Entering recovery"

        self._last_cmd = cmd
        self._last_cmd_time = now
        return cmd, debug

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._scan_history.clear()
        self._filtered_ranges = None
        self._angles = None
        self._nav_hint_theta = None
        self._dyn_barriers = []
        self._pause = False
        self._external_stop = False
        self._state = "IDLE"
        self._last_cmd = Twist()
        self._last_cmd_time = self._time()
        self._stuck_start = None
        self._stuck_flag = False
        self._recovery_active = False
        self._recovery_trial = 0
        self._recovery_phase = "back"
        self._recovery_end_time = 0.0
        self._last_gap = None

    # ------------------------------------------------------------------
    def pause(self, enabled: bool) -> None:
        self._pause = bool(enabled)

    # ------------------------------------------------------------------
    def is_stuck(self) -> bool:
        return self._stuck_flag

    # ------------------------------------------------------------------
    def current_state(self) -> str:
        return self._state

    # ------------------------------------------------------------------
    def _compute_speed(self, d_min: float, heading_mag: float) -> float:
        if d_min <= self.params.stop_distance:
            return 0.0
        if d_min <= self.params.slowdown_distance:
            span = max(1e-3, self.params.slowdown_distance - self.params.stop_distance)
            ratio = (d_min - self.params.stop_distance) / span
            base = self.params.v_min + ratio * (self.params.v_max - self.params.v_min)
        else:
            base = self.params.v_max

        curvature_factor = 1.0 / (1.0 + self.params.curvature_gain * heading_mag)
        clearance_factor = 1.0
        if d_min < self.params.min_clearance_keep:
            clearance_factor = max(0.0, d_min / max(1e-3, self.params.min_clearance_keep))
        else:
            clearance_factor = 1.0 + self.params.clearance_gain * (d_min - self.params.min_clearance_keep)
        v_cmd = base * curvature_factor
        v_cmd = max(0.0, min(self.params.v_max, v_cmd * clearance_factor))
        return v_cmd

    # ------------------------------------------------------------------
    def _apply_dynamic_inflation(self, ranges: np.ndarray) -> None:
        if self._angles is None or not self._dyn_barriers:
            return
        angle_min, angle_increment, count = self._scan_meta or (0.0, 1.0, len(ranges))
        for barrier in self._dyn_barriers:
            theta = float(barrier.get("theta", 0.0))
            rng = float(barrier.get("range", float("inf")))
            radius = float(barrier.get("radius", 0.0))
            radius = max(radius, self.params.proxemics_min)
            if not np.isfinite(rng) or rng <= 0.0:
                continue
            idx = int(round((theta - angle_min) / max(1e-6, angle_increment)))
            if idx < 0 or idx >= int(count):
                continue
            eff_range = max(0.0, rng - radius)
            angle_span = math.atan2(radius, max(rng, 1e-3))
            half_span = max(1, int(round(angle_span / max(1e-6, angle_increment))))
            start = max(0, idx - half_span)
            end = min(int(count) - 1, idx + half_span)
            for j in range(start, end + 1):
                ranges[j] = min(ranges[j], eff_range)

    # ------------------------------------------------------------------
    def _find_gaps(self, ranges: np.ndarray, angles: np.ndarray) -> List[GapInfo]:
        min_keep = max(self.params.min_clearance_keep, 0.05)
        valid = ranges > min_keep
        gaps: List[GapInfo] = []
        i = 0
        count = len(ranges)
        while i < count:
            if not valid[i]:
                i += 1
                continue
            start = i
            while i + 1 < count and valid[i + 1]:
                i += 1
            end = i
            gap = self._gap_info(start, end, ranges, angles)
            if gap.width >= self.params.door_min_width and gap.d_min > 0.0:
                gaps.append(gap)
            i += 1
        return gaps

    # ------------------------------------------------------------------
    def _gap_info(self, start: int, end: int, ranges: np.ndarray, angles: np.ndarray) -> GapInfo:
        r_segment = ranges[start : end + 1]
        a_segment = angles[start : end + 1]
        idx_best = int(np.argmax(r_segment))
        theta_best = float(a_segment[idx_best])
        theta_center = 0.5 * (angles[start] + angles[end])
        r_start = ranges[start]
        r_end = ranges[end]
        delta = angles[end] - angles[start]
        width = math.sqrt(
            max(0.0, r_start ** 2 + r_end ** 2 - 2 * r_start * r_end * math.cos(delta))
        )
        d_min = float(np.min(r_segment))
        return GapInfo(theta_center=theta_center, theta_best=theta_best, width=width, d_min=d_min)

    # ------------------------------------------------------------------
    def _select_gap(self, gaps: List[GapInfo], angles: np.ndarray) -> GapInfo:
        if not gaps:
            raise ValueError("No gaps to select from")
        if self._nav_hint_theta is None:
            return max(gaps, key=lambda g: (g.d_min, -abs(g.theta_center)))
        hint = self._nav_hint_theta
        return max(
            gaps,
            key=lambda g: (
                -abs(self._angle_diff(g.theta_center, hint)),
                g.d_min,
                g.width,
            ),
        )

    # ------------------------------------------------------------------
    def _apply_ttc(self, v_cmd: float, ranges: np.ndarray, angles: np.ndarray) -> Tuple[float, float, str]:
        ttc_values: List[float] = []
        note = ""
        ttc_stop = self.params.ttc_stop
        ttc_slow = self.params.ttc_slow

        for barrier in self._dyn_barriers:
            rng = float(barrier.get("range", float("inf")))
            v_rel = float(barrier.get("v_rel", 0.0)) + v_cmd
            if rng <= 0.0 or v_rel <= 1e-3:
                continue
            ttc = rng / v_rel
            ttc_values.append(ttc)
            if ttc <= ttc_stop:
                v_cmd = 0.0
                note = "Dynamic TTC stop"
            elif ttc <= ttc_slow:
                scale = (ttc - ttc_stop) / max(1e-3, ttc_slow - ttc_stop)
                target = self.params.v_min + scale * (self.params.v_max - self.params.v_min)
                v_cmd = min(v_cmd, target)
                note = "Dynamic TTC slow"

        forward_mask = np.cos(angles) > 0.1
        for rng, angle in zip(ranges[forward_mask], angles[forward_mask]):
            closing = v_cmd * math.cos(angle)
            if rng <= 0.0 or closing <= 1e-3:
                continue
            ttc = rng / closing
            ttc_values.append(ttc)
            if ttc <= ttc_stop:
                v_cmd = 0.0
                note = "Static TTC stop"
            elif ttc <= ttc_slow:
                scale = (ttc - ttc_stop) / max(1e-3, ttc_slow - ttc_stop)
                target = self.params.v_min + scale * (self.params.v_max - self.params.v_min)
                v_cmd = min(v_cmd, target)
                note = "Static TTC slow"

        ttc_min = min(ttc_values) if ttc_values else float("inf")
        return ttc_min, max(0.0, min(self.params.v_max, v_cmd)), note

    # ------------------------------------------------------------------
    def _limit_rate(self, value: float, prev: float, limit: float, dt: float) -> float:
        max_delta = limit * dt
        return max(prev - max_delta, min(prev + max_delta, value))

    # ------------------------------------------------------------------
    def _check_stuck(self, now: float, v_cmd: float) -> None:
        moving = self._robot_speed > self.params.stuck_vel_eps
        expecting_move = v_cmd > self.params.stuck_vel_eps
        if expecting_move and not moving:
            if self._stuck_start is None:
                self._stuck_start = now
            elif now - self._stuck_start >= self.params.stuck_time:
                self._stuck_flag = True
        else:
            self._stuck_start = None
            self._stuck_flag = False

    # ------------------------------------------------------------------
    def _start_recovery(self, now: float) -> None:
        self._recovery_active = True
        self._recovery_phase = "back"
        self._recovery_spin_dir *= -1
        speed = max(self.params.v_min, 0.05)
        duration_back = self.params.recovery_back / max(speed, 1e-3)
        self._recovery_end_time = now + duration_back
        self._state = "RECOVERY"

    # ------------------------------------------------------------------
    def _do_recovery(self, now: float) -> Tuple[Twist, str]:
        cmd = Twist()
        note = "Recovery"
        if self._recovery_phase == "back":
            cmd.linear.x = -max(self.params.v_min, 0.05)
            if now >= self._recovery_end_time:
                self._recovery_phase = "spin"
                spin_speed = max(0.2, 0.6 * self.params.w_max)
                duration_spin = self.params.recovery_spin / max(spin_speed, 1e-3)
                self._recovery_end_time = now + duration_spin
        elif self._recovery_phase == "spin":
            cmd.angular.z = self._recovery_spin_dir * max(0.2, 0.6 * self.params.w_max)
            if now >= self._recovery_end_time:
                self._recovery_active = False
                self._stuck_flag = False
                self._stuck_start = None
                self._recovery_trial += 1
                if self._recovery_trial >= self.params.recovery_trials:
                    self._state = "RECOVERY_FAILED"
                    note = "Recovery exhausted"
                else:
                    self._state = "CRUISE"
                    note = "Recovery complete"
        return cmd, note

    # ------------------------------------------------------------------
    def _gap_distance(self, theta: float, ranges: np.ndarray, angles: np.ndarray) -> float:
        idx = int(round((theta - angles[0]) / max(1e-6, angles[1] - angles[0])))
        idx = max(0, min(len(ranges) - 1, idx))
        return float(ranges[idx])

    # ------------------------------------------------------------------
    @staticmethod
    def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    # ------------------------------------------------------------------
    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        diff = (a - b + math.pi) % (2 * math.pi) - math.pi
        return diff

    # ------------------------------------------------------------------
    @staticmethod
    def _time() -> float:
        try:
            import rospy

            return rospy.get_time()
        except Exception:
            return time.time()

    # ------------------------------------------------------------------
    @property
    def robot_speed(self) -> float:
        return self._robot_speed

    # ------------------------------------------------------------------
    @property
    def robot_pose(self) -> Tuple[float, float, float]:
        return self._robot_pose

