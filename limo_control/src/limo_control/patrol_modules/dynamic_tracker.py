from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sensor_msgs.msg import LaserScan


@dataclass
class TrackParams:
    cluster_dist: float
    min_points: int
    max_tracks: int
    match_dist: float
    vanish_time: float
    speed_thresh_moving: float
    max_age: float
    smooth_alpha: float


@dataclass
class TrackedObject:
    id: int
    range: float
    theta: float
    vx: float
    vy: float
    speed: float
    is_dynamic: bool
    last_seen: float


class DynamicTracker:
    def __init__(self, params: TrackParams):
        self.params = params
        self._last_scan: Optional[LaserScan] = None
        self._tracks: Dict[int, Dict[str, float]] = {}
        self._next_id: int = 1
        self._last_step_time: Optional[float] = None

    def update_scan(self, scan: LaserScan) -> None:
        self._last_scan = scan

    def step(self, now: float) -> List[TrackedObject]:
        if self._last_scan is None:
            return []

        dt = 0.0
        if self._last_step_time is not None:
            dt = max(1e-3, now - self._last_step_time)
        self._last_step_time = now

        points = self._extract_points(self._last_scan)
        clusters = self._cluster_points(points)

        used_tracks = set()
        updated_tracks: Dict[int, Dict[str, float]] = {}

        for cluster in clusters:
            cx, cy = self._cluster_centroid(cluster)
            best_track_id = self._match_track(cx, cy, dt, used_tracks)
            if best_track_id is None and len(self._tracks) < self.params.max_tracks:
                best_track_id = self._allocate_track(cx, cy, now)
            if best_track_id is None:
                continue
            track_state = self._tracks.get(best_track_id, None)
            if track_state is None:
                track_state = self._create_track_state(cx, cy, now)
            track_state = self._alpha_beta_update(track_state, cx, cy, dt)
            track_state["last_seen"] = now
            updated_tracks[best_track_id] = track_state
            used_tracks.add(best_track_id)

        for track_id, state in self._tracks.items():
            if track_id in used_tracks:
                continue
            age = now - state.get("last_seen", now)
            if age <= self.params.vanish_time and state.get("age", 0.0) <= self.params.max_age:
                state["age"] = state.get("age", 0.0) + dt
                updated_tracks[track_id] = state

        self._tracks = updated_tracks

        tracked_objects: List[TrackedObject] = []
        for track_id, state in self._tracks.items():
            rng = math.hypot(state["x"], state["y"])
            theta = math.atan2(state["y"], state["x"])
            vx = state.get("vx", 0.0)
            vy = state.get("vy", 0.0)
            speed = math.hypot(vx, vy)
            is_dynamic = speed >= self.params.speed_thresh_moving
            tracked_objects.append(
                TrackedObject(
                    id=track_id,
                    range=rng,
                    theta=theta,
                    vx=vx,
                    vy=vy,
                    speed=speed,
                    is_dynamic=is_dynamic,
                    last_seen=state.get("last_seen", now),
                )
            )
        return tracked_objects

    def _extract_points(self, scan: LaserScan) -> List[Dict[str, float]]:
        points: List[Dict[str, float]] = []
        angle = scan.angle_min
        for r in scan.ranges:
            if math.isfinite(r) and scan.range_min <= r <= scan.range_max:
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                points.append({"x": x, "y": y, "range": r, "theta": angle})
            angle += scan.angle_increment
        return points

    def _cluster_points(self, points: List[Dict[str, float]]) -> List[List[Dict[str, float]]]:
        if not points:
            return []
        clusters: List[List[Dict[str, float]]] = []
        current_cluster: List[Dict[str, float]] = [points[0]]
        for prev, curr in zip(points, points[1:]):
            dist = math.hypot(curr["x"] - prev["x"], curr["y"] - prev["y"])
            if dist <= self.params.cluster_dist:
                current_cluster.append(curr)
            else:
                if len(current_cluster) >= self.params.min_points:
                    clusters.append(current_cluster)
                current_cluster = [curr]
        if len(current_cluster) >= self.params.min_points:
            clusters.append(current_cluster)
        return clusters

    def _cluster_centroid(self, cluster: List[Dict[str, float]]) -> tuple:
        xs = [pt["x"] for pt in cluster]
        ys = [pt["y"] for pt in cluster]
        return float(np.mean(xs)), float(np.mean(ys))

    def _match_track(self, mx: float, my: float, dt: float, used_tracks) -> Optional[int]:
        best_track_id: Optional[int] = None
        best_dist = float("inf")
        for track_id, state in self._tracks.items():
            if track_id in used_tracks:
                continue
            pred_x = state["x"] + state.get("vx", 0.0) * dt
            pred_y = state["y"] + state.get("vy", 0.0) * dt
            dist = math.hypot(mx - pred_x, my - pred_y)
            if dist < self.params.match_dist and dist < best_dist:
                best_dist = dist
                best_track_id = track_id
        return best_track_id

    def _allocate_track(self, x: float, y: float, now: float) -> int:
        track_id = self._next_id
        self._next_id += 1
        self._tracks[track_id] = self._create_track_state(x, y, now)
        return track_id

    def _create_track_state(self, x: float, y: float, now: float) -> Dict[str, float]:
        return {"x": x, "y": y, "vx": 0.0, "vy": 0.0, "last_seen": now, "age": 0.0}

    def _alpha_beta_update(self, state: Dict[str, float], mx: float, my: float, dt: float) -> Dict[str, float]:
        alpha = max(0.0, min(1.0, self.params.smooth_alpha))
        beta = 0.5 * alpha
        vx = state.get("vx", 0.0)
        vy = state.get("vy", 0.0)
        px = state["x"] + vx * dt
        py = state["y"] + vy * dt
        rx = mx - px
        ry = my - py
        px += alpha * rx
        py += alpha * ry
        if dt > 1e-3:
            vx += beta * rx / dt
            vy += beta * ry / dt
        state.update({"x": px, "y": py, "vx": vx, "vy": vy})
        return state
