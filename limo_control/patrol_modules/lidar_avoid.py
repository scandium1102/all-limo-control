#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LiDAR based reactive obstacle avoidance for LIMO cobot.

This module implements a holonomic-friendly Follow-The-Gap controller with
dynamic obstacle handling and a small recovery state machine.  The logic is
written to match the public dataclass API defined in the task specification so
that it can be shared by either a standalone node or integrated directly into
``limo_patrol.py``.

The implementation focuses on practical behaviour:

* LaserScan samples are temporally and spatially filtered before being inflated
  by the robot footprint and the configured safety margin.
* Free gaps are detected by scanning for contiguous safe sectors.  The best gap
  is selected with a blend between the local best direction and an optional
  navigation hint coming from the frontier explorer.
* Time-To-Collision (TTC) checks are performed for both raw LiDAR beams and the
  dynamic objects tracked by :class:`DynamicTracker`.  The linear command is
  reduced (or stopped) when TTC thresholds are reached.
* A light-weight recovery behaviour is triggered when the robot appears to be
  stuck for longer than ``stuck_time`` while being commanded to move.

The class exposes detailed :class:`DebugInfo` that can be published on
``/avoidance_debug`` for inspection.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

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
    """Reactive LiDAR avoidance core.

    The public API is intentionally state-less; callers update sensor inputs
    through ``update_*`` methods and then call :meth:`compute_cmd` to obtain the
    command.  Internal state is kept to support filters, TTC handling and
    recovery logic.
    """

    _RECOVERY_BACK = "back"
    _RECOVERY_SPIN = "spin"

    def __init__(self, params: AvoidParams):
        self.params = params
        self._scan: Optional[LaserScan] = None
        self._filtered_ranges: Optional[List[float]] = None
        self._raw_ranges: Optional[List[float]] = None
        self._range_history: Deque[List[float]] = deque(maxlen=max(1, params.temporal_median))
        self._nav_hint: Optional[Vector3] = None
        self._dyn_barriers: List[Dict] = []
        self._external_stop = False
        self._paused = False

        self._odom: Optional[Odometry] = None
        self._odom_speed = 0.0
        self._odom_yaw = 0.0

        self._last_cmd: Twist = Twist()
        self._last_cmd_time = self._now()
        self._stuck_start: Optional[float] = None
        self._stuck_flag = False

        self._state = "idle"

        # Recovery bookkeeping
        self._recovery_active = False
        self._recovery_phase = self._RECOVERY_BACK
        self._recovery_trial = 0
        self._recovery_phase_start = self._now()

    # ------------------------------------------------------------------
    # Public setters
    def update_scan(self, scan: LaserScan) -> None:
        self._scan = scan
        processed = self._preprocess_scan(scan)
        if processed is None:
            self._filtered_ranges = None
            self._raw_ranges = None
            return
        raw_ranges, inflated = processed
        self._raw_ranges = raw_ranges
        self._range_history.append(inflated)
        if self.params.temporal_median > 1 and len(self._range_history) >= self._range_history.maxlen:
            self._filtered_ranges = [statistics.median(values) for values in zip(*self._range_history)]
        else:
            self._filtered_ranges = inflated

    def update_odom(self, odom: Odometry) -> None:
        self._odom = odom
        self._odom_speed = odom.twist.twist.linear.x
        q = odom.pose.pose.orientation
        # Avoid import from tf for lightweight dependency.
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._odom_yaw = math.atan2(siny_cosp, cosy_cosp)

    def update_nav_hint(self, hint: Optional[Vector3]) -> None:
        self._nav_hint = hint

    def ingest_dynamic_barriers(self, dyn_barriers: List[Dict]) -> None:
        self._dyn_barriers = dyn_barriers

    def set_external_emergency_stop(self, flag: bool) -> None:
        self._external_stop = flag

    def reset(self) -> None:
        self._range_history.clear()
        self._filtered_ranges = None
        self._raw_ranges = None
        self._dyn_barriers = []
        self._stuck_start = None
        self._stuck_flag = False
        self._recovery_active = False
        self._recovery_trial = 0
        self._state = "idle"

    def pause(self, enabled: bool) -> None:
        self._paused = enabled

    def is_stuck(self) -> bool:
        return self._stuck_flag

    def current_state(self) -> str:
        return self._state

    # ------------------------------------------------------------------
    # Core command computation
    def compute_cmd(self) -> Tuple[Twist, DebugInfo]:
        twist = Twist()
        debug_gap = GapInfo(theta_center=0.0, theta_best=0.0, width=0.0, d_min=float("inf"))
        debug_info = DebugInfo(
            state="idle",
            d_min=float("inf"),
            chosen_theta=0.0,
            chosen_gap=debug_gap,
            v_cmd=0.0,
            w_cmd=0.0,
            ttc_min=float("inf"),
            stuck_flag=self._stuck_flag,
            notes="no-scan",
        )

        if self._external_stop or self._paused:
            self._state = "paused" if self._paused else "estop"
            debug_info.state = self._state
            debug_info.notes = "external stop" if self._external_stop else "paused"
            return twist, debug_info

        if self._filtered_ranges is None or self._raw_ranges is None or self._scan is None:
            debug_info.notes = "waiting for scan"
            self._state = "idle"
            return twist, debug_info

        now = self._now()

        if self._recovery_active:
            twist = self._run_recovery(now)
            debug_info.state = "recovery"
            debug_info.notes = f"phase={self._recovery_phase}, trial={self._recovery_trial}"
            debug_info.v_cmd = twist.linear.x
            debug_info.w_cmd = twist.angular.z
            debug_info.chosen_theta = 0.0
            debug_info.d_min = min(self._raw_ranges)
            debug_info.ttc_min = float("inf")
            debug_info.chosen_gap = debug_gap
            debug_info.stuck_flag = self._stuck_flag
            self._last_cmd = twist
            self._last_cmd_time = now
            return twist, debug_info

        # Process free gaps
        gaps = self._detect_gaps()
        chosen_gap, theta_ref = self._choose_heading(gaps)
        debug_gap = chosen_gap

        d_min = min(self._raw_ranges) if self._raw_ranges else float("inf")

        v_cmd = self._compute_linear_speed(theta_ref, d_min, chosen_gap)
        w_cmd = self._compute_angular_speed(theta_ref)

        # TTC checks
        ttc_min = self._evaluate_ttc(v_cmd)
        if ttc_min <= self.params.ttc_stop:
            v_cmd = 0.0
            self._state = "stop"
        elif ttc_min <= self.params.ttc_slow:
            ratio = (ttc_min - self.params.ttc_stop) / max(1e-3, self.params.ttc_slow - self.params.ttc_stop)
            v_cmd = max(0.0, min(v_cmd, self.params.v_max * ratio))
            self._state = "slow"
        else:
            self._state = "cruise" if v_cmd > 0.0 else "stop"

        twist.linear.x = v_cmd
        twist.angular.z = w_cmd

        # Stuck detection (only when we try to move forward)
        self._detect_stuck(now, v_cmd)

        debug_info = DebugInfo(
            state=self._state,
            d_min=d_min,
            chosen_theta=theta_ref,
            chosen_gap=debug_gap,
            v_cmd=v_cmd,
            w_cmd=w_cmd,
            ttc_min=ttc_min,
            stuck_flag=self._stuck_flag,
            notes="ok" if gaps else "no-gap",
        )

        self._last_cmd = twist
        self._last_cmd_time = now
        return twist, debug_info

    # ------------------------------------------------------------------
    # Internal helpers
    def _now(self) -> float:
        if rospy.is_shutdown():
            return time.time()
        try:
            return rospy.Time.now().to_sec()
        except rospy.ROSInitException:
            return time.time()

    def _preprocess_scan(self, scan: LaserScan) -> Optional[Tuple[List[float], List[float]]]:
        if not scan.ranges:
            return None

        ranges = []
        for value in scan.ranges:
            if math.isinf(value) or math.isnan(value):
                ranges.append(self.params.range_max_valid)
                continue
            value = max(self.params.range_min_valid, min(self.params.range_max_valid, value))
            ranges.append(value)

        window = max(1, self.params.spatial_window)
        if window > 1:
            smoothed = []
            n = len(ranges)
            for i in range(n):
                start = max(0, i - window)
                end = min(n, i + window + 1)
                neighbourhood = ranges[start:end]
                smoothed.append(statistics.median(neighbourhood))
            ranges = smoothed

        inflation = self.params.base_radius + self.params.safety_margin
        inflated = [max(0.0, r - inflation) for r in ranges]

        # Apply dynamic obstacle inflation directly on the inflated copy.
        if self._dyn_barriers and scan.angle_increment != 0.0:
            for barrier in self._dyn_barriers:
                idx = int(round((barrier["theta"] - scan.angle_min) / scan.angle_increment))
                if idx < 0 or idx >= len(inflated):
                    continue
                clearance = inflation + barrier.get("radius", 0.0)
                allowed = max(0.0, barrier.get("range", self.params.range_min_valid) - clearance)
                angular_radius = math.atan2(barrier.get("radius", 0.0), max(0.05, barrier.get("range", 0.1)))
                spread = max(0, int(abs(angular_radius / scan.angle_increment)))
                for j in range(max(0, idx - spread), min(len(inflated), idx + spread + 1)):
                    inflated[j] = min(inflated[j], allowed)

        return ranges, inflated

    def _detect_gaps(self) -> List[GapInfo]:
        if self._filtered_ranges is None or self._scan is None:
            return []

        scan = self._scan
        ranges = self._filtered_ranges
        threshold = self.params.min_clearance_keep
        gaps: List[GapInfo] = []

        current_start = None
        current_min = float("inf")
        best_idx = None
        best_range = -float("inf")

        for i, dist in enumerate(ranges):
            free = dist > threshold
            if free:
                current_min = min(current_min, dist)
                if best_idx is None or dist > best_range:
                    best_idx = i
                    best_range = dist
                if current_start is None:
                    current_start = i
            if (not free or i == len(ranges) - 1) and current_start is not None:
                end_idx = i if free else i - 1
                gap_info = self._make_gap_info(current_start, end_idx, current_min, best_idx)
                if gap_info:
                    gaps.append(gap_info)
                current_start = None
                current_min = float("inf")
                best_idx = None
                best_range = -float("inf")

        return gaps

    def _make_gap_info(
        self,
        start_idx: int,
        end_idx: int,
        d_min: float,
        best_idx: Optional[int],
    ) -> Optional[GapInfo]:
        if self._scan is None or self._filtered_ranges is None:
            return None
        scan = self._scan
        count = len(self._filtered_ranges)
        if start_idx < 0 or end_idx >= count or start_idx > end_idx:
            return None

        start_angle = scan.angle_min + start_idx * scan.angle_increment
        end_angle = scan.angle_min + end_idx * scan.angle_increment
        theta_center = 0.5 * (start_angle + end_angle)
        if best_idx is None:
            best_idx = start_idx + (end_idx - start_idx) // 2
        theta_best = scan.angle_min + best_idx * scan.angle_increment

        avg_range = statistics.mean(self._filtered_ranges[start_idx : end_idx + 1])
        width = abs(end_angle - start_angle) * max(self.params.lookahead_distance, avg_range)

        return GapInfo(theta_center=theta_center, theta_best=theta_best, width=width, d_min=d_min)

    def _choose_heading(self, gaps: List[GapInfo]) -> Tuple[GapInfo, float]:
        if not gaps:
            default_gap = GapInfo(theta_center=0.0, theta_best=0.0, width=0.0, d_min=0.0)
            return default_gap, 0.0

        # Filter by minimum width if available
        wide_gaps = [g for g in gaps if g.width >= self.params.gap_min_length]
        candidates = wide_gaps if wide_gaps else gaps

        nav_hint_theta = 0.0
        if self._nav_hint is not None:
            nav_hint_theta = math.atan2(self._nav_hint.y, self._nav_hint.x)

        best_score = -float("inf")
        chosen = candidates[0]
        chosen_theta = 0.0

        for gap in candidates:
            theta_gap = (1.0 - self.params.center_bias) * gap.theta_best + self.params.center_bias * gap.theta_center
            theta_ref = (1.0 - self.params.goal_bias) * theta_gap + self.params.goal_bias * nav_hint_theta
            width_bonus = min(1.0, gap.width / max(0.01, self.params.door_min_width))
            score = width_bonus * (gap.d_min + 1.0) - abs(theta_ref - nav_hint_theta)
            if score > best_score:
                best_score = score
                chosen = gap
                chosen_theta = theta_ref

        return chosen, chosen_theta

    def _compute_linear_speed(self, theta_ref: float, d_min: float, gap: GapInfo) -> float:
        params = self.params
        v_cmd = params.v_max

        if d_min <= params.stop_distance:
            return 0.0
        elif d_min <= params.slowdown_distance:
            ratio = (d_min - params.stop_distance) / max(1e-3, params.slowdown_distance - params.stop_distance)
            v_cmd = params.v_max * ratio

        # Encourage higher speed for wide/clear gaps
        clearance_bonus = params.clearance_gain * max(0.0, gap.d_min - params.min_clearance_keep)
        v_cmd = min(params.v_max, v_cmd + clearance_bonus)

        curvature_penalty = 1.0 / (1.0 + params.curvature_gain * abs(theta_ref))
        v_cmd *= curvature_penalty

        if v_cmd > 0.0:
            v_cmd = max(params.v_min, min(v_cmd, params.v_max))

        return v_cmd

    def _compute_angular_speed(self, theta_ref: float) -> float:
        params = self.params
        w_cmd = params.yaw_kp * theta_ref
        w_cmd = max(-params.w_max, min(params.w_max, w_cmd))
        return w_cmd

    def _evaluate_ttc(self, v_cmd: float) -> float:
        if self._scan is None or self._raw_ranges is None:
            return float("inf")

        scan = self._scan
        min_ttc = float("inf")
        ranges = self._raw_ranges

        if v_cmd > 1e-3:
            for i, dist in enumerate(ranges):
                angle = scan.angle_min + i * scan.angle_increment
                forward_speed = v_cmd * math.cos(angle)
                if forward_speed <= 1e-3:
                    continue
                ttc = dist / max(1e-3, forward_speed)
                if ttc < min_ttc:
                    min_ttc = ttc

        # Dynamic barriers provided by tracker
        for barrier in self._dyn_barriers:
            dist = barrier.get("range", float("inf"))
            radius = barrier.get("radius", 0.0)
            closing = max(0.0, -barrier.get("v_rel", 0.0))
            if v_cmd > closing:
                closing = v_cmd
            if closing <= 1e-3:
                continue
            effective = max(0.01, dist - radius)
            ttc = effective / closing
            if ttc < min_ttc:
                min_ttc = ttc

        return min_ttc

    def _detect_stuck(self, now: float, v_cmd: float) -> None:
        actual_speed = abs(self._odom_speed)
        if v_cmd > self.params.v_min * 0.5:
            if actual_speed < self.params.stuck_vel_eps:
                if self._stuck_start is None:
                    self._stuck_start = now
                elif now - self._stuck_start > self.params.stuck_time:
                    self._stuck_flag = True
                    self._enter_recovery(now)
            else:
                self._stuck_start = None
                self._stuck_flag = False
        else:
            self._stuck_start = None
            self._stuck_flag = False

    def _enter_recovery(self, now: float) -> None:
        if self.params.recovery_trials <= 0:
            return
        if self._recovery_active and self._recovery_trial >= self.params.recovery_trials:
            return
        self._recovery_active = True
        if not self._recovery_trial:
            self._recovery_trial = 0
        self._recovery_phase = self._RECOVERY_BACK
        self._recovery_phase_start = now
        self._recovery_trial += 1

    def _run_recovery(self, now: float) -> Twist:
        twist = Twist()
        back_speed = -abs(self.params.v_min)
        spin_speed = self.params.w_max * 0.6
        back_time = abs(self.params.recovery_back / back_speed) if back_speed != 0 else 0.0
        spin_time = abs(self.params.recovery_spin / spin_speed) if spin_speed != 0 else 0.0

        if self._recovery_phase == self._RECOVERY_BACK:
            twist.linear.x = back_speed
            if now - self._recovery_phase_start >= back_time:
                self._recovery_phase = self._RECOVERY_SPIN
                self._recovery_phase_start = now
        elif self._recovery_phase == self._RECOVERY_SPIN:
            direction = 1 if (self._recovery_trial % 2 == 1) else -1
            twist.angular.z = direction * spin_speed
            if now - self._recovery_phase_start >= spin_time:
                if self._recovery_trial >= self.params.recovery_trials:
                    self._recovery_active = False
                    self._stuck_flag = False
                else:
                    self._recovery_phase = self._RECOVERY_BACK
                    self._recovery_phase_start = now
        else:
            self._recovery_active = False

        if not self._recovery_active:
            twist = Twist()

        return twist


__all__ = [
    "AvoidParams",
    "GapInfo",
    "DebugInfo",
    "LidarAvoider",
]

