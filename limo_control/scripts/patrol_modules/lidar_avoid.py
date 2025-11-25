# Copyright 2024
#
# Lidar based avoidance module for LIMO cobot with mecanum wheels.
# The module implements follow-the-gap navigation with dynamic obstacle
# awareness and simple recovery behaviour.

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

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
    method: str
    goal_bias: float
    center_bias: float
    gap_min_length: float
    lookahead_distance: float
    yaw_kp: float
    curvature_gain: float
    clearance_gain: float
    min_clearance_keep: float
    narrow_hint_angle: float
    narrow_extra_clearance: float
    corridor_lock_width: float
    corridor_lock_time: float
    heading_hysteresis: float
    heading_lpf_alpha: float
    median_window: int
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
    chosen_gap: GapInfo
    v_cmd: float
    w_cmd: float
    ttc_min: float
    stuck_flag: bool
    notes: str


class LidarAvoider:
    """Follow-the-gap based reactive navigator with TTC control."""

    def __init__(self, params: AvoidParams):
        self.params = params
        self._scan_queue: Deque[np.ndarray] = deque(maxlen=max(1, params.temporal_median))
        self._last_scan: Optional[LaserScan] = None
        self._last_ranges: Optional[np.ndarray] = None
        self._nav_hint: Optional[Vector3] = None
        self._dyn_barriers: List[Dict] = []
        self._external_stop: bool = False
        self._paused: bool = False
        self._last_cmd = Twist()
        self._last_cmd_time: Optional[float] = None
        self._state: str = "IDLE"
        self._current_speed: float = 0.0
        self._current_yaw_rate: float = 0.0
        self._last_motion_time: Optional[float] = None
        self._stuck_flag: bool = False
        self._recovery_active: bool = False
        self._recovery_start: float = 0.0
        self._recovery_phase: str = ""
        self._recovery_trial: int = 0
        self._recovery_sign: int = 1
        self._stuck_failed: bool = False
        self._theta_lp: Optional[float] = None
        self._theta_hold: float = 0.0
        self._corridor_lock_until: float = 0.0
        default_hyst = getattr(self.params, "heading_hysteresis", 0.12)
        default_alpha = getattr(self.params, "heading_lpf_alpha", 0.4)
        self._hyst_rad = float(rospy.get_param("~avoid_params/heading_hysteresis", default_hyst))
        self._alpha_theta = float(rospy.get_param("~avoid_params/heading_lpf_alpha", default_alpha))
        self._hyst_rad = max(0.0, self._hyst_rad)
        self._alpha_theta = max(0.0, min(1.0, self._alpha_theta))
        self.params.heading_hysteresis = self._hyst_rad
        self.params.heading_lpf_alpha = self._alpha_theta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_scan(self, scan: LaserScan) -> None:
        self._last_scan = scan
        ranges = self._preprocess_scan(scan)
        self._scan_queue.append(ranges)
        if len(self._scan_queue) == 1:
            self._last_ranges = ranges
        else:
            stacked = np.stack(list(self._scan_queue), axis=0)
            self._last_ranges = np.median(stacked, axis=0)

    def update_odom(self, odom: Odometry) -> None:
        twist = odom.twist.twist
        if self.params.holonomic:
            linear = math.hypot(twist.linear.x, twist.linear.y)
        else:
            linear = twist.linear.x
        self._current_speed = linear
        self._current_yaw_rate = twist.angular.z
        now = odom.header.stamp.to_sec() if odom.header.stamp else rospy.Time.now().to_sec()
        if abs(linear) > self.params.stuck_vel_eps:
            self._last_motion_time = now
            self._stuck_flag = False

    def update_nav_hint(self, hint: Optional[Vector3]) -> None:
        self._nav_hint = hint

    def ingest_dynamic_barriers(self, dyn_barriers: List[Dict]) -> None:
        self._dyn_barriers = dyn_barriers

    def set_external_emergency_stop(self, flag: bool) -> None:
        self._external_stop = flag

    def compute_cmd(self) -> Tuple[Twist, DebugInfo]:
        now = rospy.Time.now().to_sec()
        dt = 0.0 if self._last_cmd_time is None else max(1e-3, now - self._last_cmd_time)
        notes: List[str] = []

        if self._external_stop:
            self._state = "EMERGENCY_STOP"
            cmd = Twist()
            dbg = self._make_debug(cmd, 0.0, 0.0, float("inf"), notes)
            self._last_cmd_time = now
            self._last_cmd = cmd
            return cmd, dbg

        if self._paused:
            self._state = "PAUSED"
            cmd = Twist()
            dbg = self._make_debug(cmd, 0.0, 0.0, float("inf"), notes)
            self._last_cmd_time = now
            self._last_cmd = cmd
            return cmd, dbg

        if self._last_ranges is None or self._last_scan is None:
            self._state = "NO_SCAN"
            cmd = Twist()
            dbg = self._make_debug(cmd, 0.0, 0.0, float("inf"), ["waiting_for_scan"])
            self._last_cmd_time = now
            self._last_cmd = cmd
            return cmd, dbg

        ranges = np.copy(self._last_ranges)
        angles = self._beam_angles(self._last_scan)

        blocked_mask = self._build_blocked_mask(ranges, angles)
        gap = self._select_gap(ranges, angles, blocked_mask)

        d_min = float(np.min(ranges)) if ranges.size > 0 else float("inf")

        theta_hint = self._nav_hint_angle()
        chosen_theta = gap.theta_best
        blended_theta = self._blend_theta(chosen_theta, gap.theta_center, theta_hint)
        if gap.width > 0.0 and gap.width <= self.params.corridor_lock_width:
            self._corridor_lock_until = max(
                self._corridor_lock_until, now + self.params.corridor_lock_time
            )
        elif now > self._corridor_lock_until:
            self._corridor_lock_until = now

        theta_cmd = self._smooth_heading(blended_theta, now)

        w_cmd = self._compute_angular_cmd(theta_cmd)
        v_cmd = self._compute_linear_cmd(d_min, gap.d_min)

        ttc_min = self._compute_ttc(ranges, angles, v_cmd)
        if math.isfinite(ttc_min):
            if ttc_min <= self.params.ttc_stop:
                v_cmd = 0.0
                notes.append("ttc_stop")
                self._state = "STOP"
            elif ttc_min <= self.params.ttc_slow:
                ratio = (ttc_min - self.params.ttc_stop) / max(1e-3, self.params.ttc_slow - self.params.ttc_stop)
                v_cmd = min(v_cmd, self.params.v_min + ratio * (self.params.v_max - self.params.v_min))
                notes.append("ttc_slow")
                self._state = "SLOW"
        else:
            self._state = "CRUISE"

        if d_min <= self.params.stop_distance:
            v_cmd = 0.0
            self._state = "STOP"
            notes.append("proximity_stop")
        elif d_min <= self.params.slowdown_distance and self._state != "STOP":
            self._state = "SLOW"

        v_cmd = self._apply_curvature_clearance(v_cmd, w_cmd, gap.d_min)

        if self._check_stuck(now, v_cmd):
            notes.append("stuck_detected")
            if not self._recovery_active and self._recovery_trial < self.params.recovery_trials:
                self._start_recovery(now)

        if self._recovery_active:
            cmd = self._run_recovery(now)
            v_cmd = cmd.linear.x
            w_cmd = cmd.angular.z
            notes.append("recovery")
            if not self._recovery_active and self._stuck_flag:
                notes.append("recovery_failed")
        else:
            cmd = Twist()
            cmd.linear.x = v_cmd
            cmd.angular.z = w_cmd
            if self.params.holonomic and self._nav_hint is not None:
                cmd.linear.y = max(min(self._nav_hint.y, self.params.v_max), -self.params.v_max)

        cmd = self._limit_acceleration(cmd, dt)

        dbg = self._make_debug(cmd, d_min, theta_cmd, ttc_min, notes, gap)

        self._last_cmd_time = now
        self._last_cmd = cmd
        return cmd, dbg

    def reset(self) -> None:
        self._scan_queue.clear()
        self._last_scan = None
        self._last_ranges = None
        self._nav_hint = None
        self._dyn_barriers = []
        self._paused = False
        self._state = "IDLE"
        self._last_cmd = Twist()
        self._last_cmd_time = None
        self._stuck_flag = False
        self._recovery_active = False
        self._recovery_trial = 0
        self._stuck_failed = False
        self._theta_lp = None
        self._theta_hold = 0.0
        self._corridor_lock_until = 0.0

    def pause(self, enabled: bool) -> None:
        self._paused = enabled

    def is_stuck(self) -> bool:
        return self._stuck_failed

    def current_state(self) -> str:
        return self._state

    def current_speed(self) -> float:
        return self._current_speed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _preprocess_scan(self, scan: LaserScan) -> np.ndarray:
        ranges = np.array(scan.ranges, dtype=float)
        ranges[~np.isfinite(ranges)] = scan.range_max
        ranges = np.clip(ranges, self.params.range_min_valid, self.params.range_max_valid)

        if self.params.spatial_window > 1:
            kernel = np.ones(self.params.spatial_window) / float(self.params.spatial_window)
            ranges = np.convolve(ranges, kernel, mode="same")

        inflate = self.params.base_radius + self.params.safety_margin
        ranges = np.maximum(0.0, ranges - inflate)
        return ranges

    def _beam_angles(self, scan: LaserScan) -> np.ndarray:
        count = len(scan.ranges)
        return scan.angle_min + np.arange(count) * scan.angle_increment

    def _build_blocked_mask(self, ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
        clearance = ranges < self.params.min_clearance_keep
        blocked = clearance.copy()
        if self._dyn_barriers:
            for barrier in self._dyn_barriers:
                dist = max(barrier.get("range", 0.0), 1e-3)
                theta = barrier.get("theta", 0.0)
                radius = max(self.params.proxemics_min, barrier.get("radius", 0.0))
                half_angle = math.asin(max(0.0, min(1.0, radius / dist))) if dist > radius else math.pi / 2.0
                blocked |= np.abs(self._wrap_angle(angles - theta)) <= half_angle
        blocked |= ranges < self.params.base_radius * 0.2
        return blocked

    def _select_gap(self, ranges: np.ndarray, angles: np.ndarray, blocked_mask: np.ndarray) -> GapInfo:
        free_mask = ~blocked_mask
        best_gap = GapInfo(theta_center=0.0, theta_best=0.0, width=0.0, d_min=float(np.min(ranges)))
        start_idx: Optional[int] = None

        for idx, free in enumerate(free_mask):
            if free and start_idx is None:
                start_idx = idx
            elif (not free or idx == len(free_mask) - 1) and start_idx is not None:
                end_idx = idx if free and idx == len(free_mask) - 1 else idx - 1
                gap = self._evaluate_gap(start_idx, end_idx, ranges, angles)
                start_idx = None
                if gap and self._is_better_gap(gap, best_gap):
                    best_gap = gap

        if best_gap.width == 0.0:
            max_idx = int(np.argmax(ranges))
            best_gap = GapInfo(
                theta_center=float(angles[max_idx]),
                theta_best=float(angles[max_idx]),
                width=0.0,
                d_min=float(np.min(ranges)),
            )
        return best_gap

    def _evaluate_gap(
        self,
        start_idx: int,
        end_idx: int,
        ranges: np.ndarray,
        angles: np.ndarray,
    ) -> Optional[GapInfo]:
        if end_idx <= start_idx:
            return None

        sub_ranges = ranges[start_idx : end_idx + 1]
        sub_angles = angles[start_idx : end_idx + 1]
        if sub_ranges.size == 0:
            return None

        min_range = float(np.min(sub_ranges))
        width_angle = float(abs(sub_angles[-1] - sub_angles[0]))
        representative_range = float(min(np.max(sub_ranges), self.params.lookahead_distance))
        width_m = representative_range * width_angle

        min_width = max(self.params.gap_min_length, self.params.door_min_width)
        allow_narrow = False
        if width_m < min_width:
            hint = self._nav_hint_angle()
            theta_gap = float(0.5 * (sub_angles[0] + sub_angles[-1]))
            min_passable = 2.0 * (self.params.base_radius + self.params.safety_margin) + self.params.narrow_extra_clearance
            if (
                hint is not None
                and width_m >= min_passable
                and abs(self._wrap_angle(theta_gap - hint)) <= self.params.narrow_hint_angle
            ):
                allow_narrow = True
        if width_m < min_width and not allow_narrow:
            return None

        best_idx_local = int(np.argmax(sub_ranges))
        theta_best = float(sub_angles[best_idx_local])
        theta_center = float(0.5 * (sub_angles[0] + sub_angles[-1]))

        return GapInfo(theta_center=theta_center, theta_best=theta_best, width=width_m, d_min=min_range)

    def _is_better_gap(self, new_gap: GapInfo, current_best: GapInfo) -> bool:
        if new_gap is None:
            return False
        if current_best.width == 0.0:
            return True
        hint = self._nav_hint_angle()
        if hint is None:
            hint = 0.0
        current_bias = abs(self._wrap_angle(current_best.theta_best - hint))
        new_bias = abs(self._wrap_angle(new_gap.theta_best - hint))
        if math.isclose(new_gap.width, current_best.width, rel_tol=0.2):
            return new_bias < current_bias
        return new_gap.width > current_best.width

    def _nav_hint_angle(self) -> Optional[float]:
        if self._nav_hint is None:
            return None
        return math.atan2(self._nav_hint.y, self._nav_hint.x)

    def _blend_theta(self, theta_best: float, theta_center: float, theta_hint: Optional[float]) -> float:
        theta_gap = (1.0 - self.params.center_bias) * theta_best + self.params.center_bias * theta_center
        if theta_hint is None:
            return self._wrap_angle(theta_gap)
        blended = (1.0 - self.params.goal_bias) * theta_gap + self.params.goal_bias * theta_hint
        return self._wrap_angle(blended)

    def _smooth_heading(self, theta: float, now: float) -> float:
        if self._theta_lp is None:
            self._theta_lp = theta
        else:
            delta = self._wrap_angle(theta - self._theta_lp)
            self._theta_lp = self._wrap_angle(self._theta_lp + self._alpha_theta * delta)
        candidate = self._theta_lp
        if now <= self._corridor_lock_until:
            candidate = self._theta_hold
        if abs(self._wrap_angle(candidate - self._theta_hold)) > self._hyst_rad:
            self._theta_hold = candidate
        return self._theta_hold

    def _compute_angular_cmd(self, theta_ref: float) -> float:
        w_cmd = self.params.yaw_kp * theta_ref
        return float(np.clip(w_cmd, -self.params.w_max, self.params.w_max))

    def _compute_linear_cmd(self, d_min: float, gap_clearance: float) -> float:
        if d_min <= self.params.stop_distance:
            return 0.0

        if d_min <= self.params.slowdown_distance:
            ratio = (d_min - self.params.stop_distance) / max(1e-3, self.params.slowdown_distance - self.params.stop_distance)
            v = self.params.v_min + ratio * (self.params.v_max - self.params.v_min)
        else:
            v = self.params.v_max

        clearance_boost = max(0.0, gap_clearance - self.params.min_clearance_keep)
        v *= 1.0 + self.params.clearance_gain * clearance_boost
        return min(v, self.params.v_max)

    def _apply_curvature_clearance(self, v_cmd: float, w_cmd: float, clearance: float) -> float:
        if abs(w_cmd) > 1e-3 and v_cmd > 0.0:
            curvature_penalty = 1.0 / (1.0 + self.params.curvature_gain * abs(w_cmd))
            v_cmd *= curvature_penalty
        if clearance < self.params.min_clearance_keep:
            v_cmd *= max(0.1, clearance / max(1e-3, self.params.min_clearance_keep))
        return max(0.0, min(self.params.v_max, v_cmd))

    def _compute_ttc(self, ranges: np.ndarray, angles: np.ndarray, v_cmd: float) -> float:
        speed = max(self._current_speed, v_cmd)
        if speed <= 1e-3:
            return float("inf")

        ttc_values: List[float] = []
        for dist, ang in zip(ranges, angles):
            closing = speed * max(0.0, math.cos(ang))
            if closing > 1e-3:
                ttc_values.append(dist / closing)

        for barrier in self._dyn_barriers:
            dist = barrier.get("range", 0.0)
            v_rel = barrier.get("v_rel", 0.0)
            closing = max(0.0, -v_rel)
            if closing <= 1e-3:
                continue
            ttc_values.append(dist / closing)

        if not ttc_values:
            return float("inf")
        return min(ttc_values)

    def _check_stuck(self, now: float, v_cmd: float) -> bool:
        if v_cmd <= self.params.v_min:
            return False
        if self._last_motion_time is None:
            self._last_motion_time = now
            return False
        if now - self._last_motion_time > self.params.stuck_time:
            self._stuck_flag = True
            if self._recovery_trial >= self.params.recovery_trials:
                self._stuck_failed = True
            return True
        return False

    def _start_recovery(self, now: float) -> None:
        self._recovery_active = True
        self._recovery_phase = "back"
        self._recovery_start = now
        self._recovery_trial += 1
        self._recovery_sign = 1 if self._recovery_trial % 2 == 0 else -1

    def _run_recovery(self, now: float) -> Twist:
        cmd = Twist()
        elapsed = now - self._recovery_start
        if self._recovery_phase == "back":
            duration = self.params.recovery_back / max(self.params.v_min, 1e-2)
            cmd.linear.x = -self.params.v_min
            if elapsed >= duration:
                self._recovery_phase = "spin"
                self._recovery_start = now
        elif self._recovery_phase == "spin":
            target_w = self._recovery_sign * 0.7 * self.params.w_max
            duration = abs(self.params.recovery_spin) / max(abs(target_w), 1e-3)
            cmd.angular.z = target_w
            if elapsed >= duration:
                self._recovery_active = False
                self._stuck_flag = False
                self._last_motion_time = now
        else:
            self._recovery_phase = "back"
            self._recovery_start = now
        return cmd

    def _limit_acceleration(self, cmd: Twist, dt: float) -> Twist:
        if dt <= 0.0:
            return cmd
        max_dv = self.params.ax_max * dt
        max_dw = self.params.aw_max * dt

        dv = cmd.linear.x - self._last_cmd.linear.x
        if abs(dv) > max_dv:
            cmd.linear.x = self._last_cmd.linear.x + math.copysign(max_dv, dv)

        dw = cmd.angular.z - self._last_cmd.angular.z
        if abs(dw) > max_dw:
            cmd.angular.z = self._last_cmd.angular.z + math.copysign(max_dw, dw)

        if self.params.holonomic:
            dy = cmd.linear.y - self._last_cmd.linear.y
            if abs(dy) > max_dv:
                cmd.linear.y = self._last_cmd.linear.y + math.copysign(max_dv, dy)
        else:
            cmd.linear.y = 0.0
        return cmd

    def _make_debug(
        self,
        cmd: Twist,
        d_min: float,
        theta: float,
        ttc: float,
        notes: List[str],
        gap: Optional[GapInfo] = None,
    ) -> DebugInfo:
        if gap is None:
            gap = GapInfo(theta_center=0.0, theta_best=0.0, width=0.0, d_min=d_min)
        notes_str = ",".join(sorted(set(notes))) if notes else ""
        return DebugInfo(
            state=self._state,
            d_min=d_min,
            chosen_theta=theta,
            chosen_gap=gap,
            v_cmd=cmd.linear.x,
            w_cmd=cmd.angular.z,
            ttc_min=ttc,
            stuck_flag=self._stuck_flag or self._stuck_failed,
            notes=notes_str,
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def to_json(self, debug: DebugInfo) -> str:
        gap_dict = {
            "theta_center": debug.chosen_gap.theta_center,
            "theta_best": debug.chosen_gap.theta_best,
            "width": debug.chosen_gap.width,
            "d_min": debug.chosen_gap.d_min,
        }
        payload = {
            "state": debug.state,
            "d_min": debug.d_min,
            "chosen_theta": debug.chosen_theta,
            "gap": gap_dict,
            "v_cmd": debug.v_cmd,
            "w_cmd": debug.w_cmd,
            "ttc_min": debug.ttc_min,
            "stuck_flag": debug.stuck_flag,
            "notes": debug.notes,
        }
        return json.dumps(payload)
