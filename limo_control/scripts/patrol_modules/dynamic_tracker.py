"""Dynamic obstacle tracking from 2D LiDAR scans."""

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
    """Simple cluster-based tracker using an alpha-beta like filter."""

    def __init__(self, params: TrackParams):
        self.params = params
        self._scan: Optional[LaserScan] = None
        self._tracks: List[Dict] = []
        self._next_id = 1

    # ------------------------------------------------------------------
    def update_scan(self, scan: LaserScan) -> None:
        self._scan = scan

    # ------------------------------------------------------------------
    def step(self, now: float) -> List[TrackedObject]:
        if self._scan is None or not self._scan.ranges:
            return []

        clusters = self._cluster_points(self._scan)
        measurements = self._cluster_centroids(clusters)
        self._update_tracks(measurements, now)
        self._prune_tracks(now)
        return [
            TrackedObject(
                id=track["id"],
                range=math.hypot(track["x"], track["y"]),
                theta=math.atan2(track["y"], track["x"]),
                vx=track["vx"],
                vy=track["vy"],
                speed=math.hypot(track["vx"], track["vy"]),
                is_dynamic=track["dynamic"],
                last_seen=track["last_seen"],
            )
            for track in self._tracks
        ]

    # ------------------------------------------------------------------
    def _cluster_points(self, scan: LaserScan) -> List[List[Dict[str, float]]]:
        clusters: List[List[Dict[str, float]]] = []
        current: List[Dict[str, float]] = []
        prev_point: Optional[Dict[str, float]] = None
        for idx, rng in enumerate(scan.ranges):
            if not math.isfinite(rng):
                prev_point = None
                if current:
                    clusters.append(current)
                    current = []
                continue
            angle = scan.angle_min + idx * scan.angle_increment
            x = rng * math.cos(angle)
            y = rng * math.sin(angle)
            point = {"x": x, "y": y, "range": rng, "theta": angle}
            if prev_point is not None:
                dist = math.hypot(point["x"] - prev_point["x"], point["y"] - prev_point["y"])
                if dist > self.params.cluster_dist:
                    if current:
                        clusters.append(current)
                    current = [point]
                else:
                    current.append(point)
            else:
                current = [point]
            prev_point = point
        if current:
            clusters.append(current)
        return clusters

    # ------------------------------------------------------------------
    def _cluster_centroids(self, clusters: List[List[Dict[str, float]]]) -> List[Dict[str, float]]:
        measurements: List[Dict[str, float]] = []
        for pts in clusters:
            if len(pts) < self.params.min_points:
                continue
            xs = np.array([p["x"] for p in pts])
            ys = np.array([p["y"] for p in pts])
            x = float(np.mean(xs))
            y = float(np.mean(ys))
            rng = math.hypot(x, y)
            theta = math.atan2(y, x)
            measurements.append({"x": x, "y": y, "range": rng, "theta": theta})
        return measurements

    # ------------------------------------------------------------------
    def _update_tracks(self, measurements: List[Dict[str, float]], now: float) -> None:
        alpha = min(max(self.params.smooth_alpha, 0.0), 1.0)

        # Predict
        for track in self._tracks:
            dt = max(0.0, now - track["last_update"])
            track["x"] += track["vx"] * dt
            track["y"] += track["vy"] * dt
            track["age"] += dt

        # Association (greedy)
        unmatched_measurements = measurements[:]
        for track in self._tracks:
            best_idx = -1
            best_dist = float("inf")
            for idx, meas in enumerate(unmatched_measurements):
                dist = math.hypot(track["x"] - meas["x"], track["y"] - meas["y"])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx >= 0 and best_dist <= self.params.match_dist:
                meas = unmatched_measurements.pop(best_idx)
                dt = max(1e-3, now - track["last_update"])
                x_pred = track["x"]
                y_pred = track["y"]
                track["x"] = (1.0 - alpha) * x_pred + alpha * meas["x"]
                track["y"] = (1.0 - alpha) * y_pred + alpha * meas["y"]
                vx_meas = (meas["x"] - track["last_meas_x"]) / dt
                vy_meas = (meas["y"] - track["last_meas_y"]) / dt
                track["vx"] = (1.0 - alpha) * track["vx"] + alpha * vx_meas
                track["vy"] = (1.0 - alpha) * track["vy"] + alpha * vy_meas
                track["last_meas_x"] = meas["x"]
                track["last_meas_y"] = meas["y"]
                track["last_seen"] = now
                track["last_update"] = now
                track["dynamic"] = math.hypot(track["vx"], track["vy"]) > self.params.speed_thresh_moving
            else:
                track["dynamic"] = math.hypot(track["vx"], track["vy"]) > self.params.speed_thresh_moving

        # Create new tracks
        for meas in unmatched_measurements:
            if len(self._tracks) >= self.params.max_tracks:
                break
            track = {
                "id": self._next_id,
                "x": meas["x"],
                "y": meas["y"],
                "vx": 0.0,
                "vy": 0.0,
                "last_meas_x": meas["x"],
                "last_meas_y": meas["y"],
                "last_seen": now,
                "last_update": now,
                "age": 0.0,
                "dynamic": False,
            }
            self._tracks.append(track)
            self._next_id += 1

    # ------------------------------------------------------------------
    def _prune_tracks(self, now: float) -> None:
        remaining: List[Dict] = []
        for track in self._tracks:
            if now - track["last_seen"] > self.params.vanish_time:
                continue
            if track["age"] > self.params.max_age:
                continue
            remaining.append(track)
        self._tracks = remaining

