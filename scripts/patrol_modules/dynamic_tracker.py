"""Dynamic obstacle tracking using 2D LiDAR clusters."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sensor_msgs.msg import LaserScan


@dataclass
class TrackParams:
    cluster_dist: float          # m，光束鄰近聚類閾值
    min_points: int
    max_tracks: int
    match_dist: float            # m，跨幀關聯門檻
    vanish_time: float           # s，失配多久移除
    speed_thresh_moving: float   # m/s，判定「動態」
    max_age: float               # s，軌跡最長壽命
    smooth_alpha: float          # 0~1，α-β濾波


@dataclass
class TrackedObject:
    id: int
    range: float
    theta: float
    vx: float
    vy: float         # 在 base_link 座標
    speed: float
    is_dynamic: bool
    last_seen: float


class DynamicTracker:
    """Cluster LaserScan returns and track moving objects with a simple α-β filter."""

    def __init__(self, params: TrackParams):
        self.params = params
        self._scan: Optional[LaserScan] = None
        self._tracks: List[dict] = []
        self._next_id = 1

    # ------------------------------------------------------------------
    def update_scan(self, scan: LaserScan) -> None:
        self._scan = scan

    # ------------------------------------------------------------------
    def step(self, now: float) -> List[TrackedObject]:
        if self._scan is None:
            return []

        detections = self._cluster_scan(self._scan)
        assigned = set()
        alpha = max(0.0, min(1.0, self.params.smooth_alpha))
        beta = 0.5 * alpha

        for det in detections:
            best_idx = None
            best_dist = self.params.match_dist
            det_x, det_y = det["x"], det["y"]
            for idx, track in enumerate(self._tracks):
                if idx in assigned:
                    continue
                dt = max(1e-3, now - track["last_seen"])
                pred_x = track["x"] + track["vx"] * dt
                pred_y = track["y"] + track["vy"] * dt
                dist = math.hypot(det_x - pred_x, det_y - pred_y)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is None:
                if len(self._tracks) >= self.params.max_tracks:
                    continue
                self._tracks.append(
                    {
                        "id": self._next_id,
                        "x": det_x,
                        "y": det_y,
                        "vx": 0.0,
                        "vy": 0.0,
                        "last_seen": now,
                        "created": now,
                    }
                )
                self._next_id += 1
                assigned.add(len(self._tracks) - 1)
                continue

            track = self._tracks[best_idx]
            dt = max(1e-3, now - track["last_seen"])
            pred_x = track["x"] + track["vx"] * dt
            pred_y = track["y"] + track["vy"] * dt
            resid_x = det_x - pred_x
            resid_y = det_y - pred_y
            track["x"] = pred_x + alpha * resid_x
            track["y"] = pred_y + alpha * resid_y
            track["vx"] = track["vx"] + (beta / dt) * resid_x
            track["vy"] = track["vy"] + (beta / dt) * resid_y
            track["last_seen"] = now
            assigned.add(best_idx)

        # Propagate unassigned tracks forward in time to keep predictions fresh
        for idx, track in enumerate(self._tracks):
            if idx in assigned:
                continue
            dt = max(0.0, now - track["last_seen"])
            track["x"] += track["vx"] * dt
            track["y"] += track["vy"] * dt
            track["last_seen"] = now

        self._prune_tracks(now)
        return [self._to_msg(track) for track in self._tracks]

    # ------------------------------------------------------------------
    def _cluster_scan(self, scan: LaserScan) -> List[dict]:
        ranges = np.array(scan.ranges, dtype=float)
        angle = scan.angle_min
        angle_inc = scan.angle_increment
        valid_mask = np.logical_and(ranges >= scan.range_min, ranges <= scan.range_max)

        clusters: List[List[tuple]] = []
        current: List[tuple] = []
        prev_point: Optional[tuple] = None

        for idx, valid in enumerate(valid_mask):
            if not valid:
                if len(current) >= self.params.min_points:
                    clusters.append(current)
                current = []
                prev_point = None
                angle += angle_inc
                continue

            rng = ranges[idx]
            x = rng * math.cos(angle)
            y = rng * math.sin(angle)
            point = (x, y, rng, angle)

            if prev_point is not None:
                dist = math.hypot(x - prev_point[0], y - prev_point[1])
                if dist > self.params.cluster_dist:
                    if len(current) >= self.params.min_points:
                        clusters.append(current)
                    current = []
            current.append(point)
            prev_point = point
            angle += angle_inc

        if len(current) >= self.params.min_points:
            clusters.append(current)

        detections: List[dict] = []
        for cluster in clusters:
            pts = np.array(cluster)
            mean_xy = pts[:, :2].mean(axis=0)
            mean_range = float(np.linalg.norm(mean_xy))
            theta = math.atan2(mean_xy[1], mean_xy[0])
            detections.append({"x": mean_xy[0], "y": mean_xy[1], "range": mean_range, "theta": theta})

        return detections

    # ------------------------------------------------------------------
    def _prune_tracks(self, now: float) -> None:
        new_tracks: List[dict] = []
        for track in self._tracks:
            age = now - track["created"]
            if age > self.params.max_age:
                continue
            if now - track["last_seen"] > self.params.vanish_time:
                continue
            new_tracks.append(track)
        self._tracks = new_tracks

    # ------------------------------------------------------------------
    def _to_msg(self, track: dict) -> TrackedObject:
        rng = math.hypot(track["x"], track["y"])
        theta = math.atan2(track["y"], track["x"])
        speed = math.hypot(track["vx"], track["vy"])
        is_dynamic = speed > self.params.speed_thresh_moving
        return TrackedObject(
            id=track["id"],
            range=rng,
            theta=theta,
            vx=track["vx"],
            vy=track["vy"],
            speed=speed,
            is_dynamic=is_dynamic,
            last_seen=track["last_seen"],
        )

