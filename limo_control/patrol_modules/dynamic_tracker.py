#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic obstacle tracker for 2D LiDAR clusters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    """Cluster-based dynamic obstacle tracker."""

    def __init__(self, params: TrackParams):
        self.params = params
        self._scan: Optional[LaserScan] = None
        self._tracks: Dict[int, TrackedObject] = {}
        self._next_id = 1

    def update_scan(self, scan: LaserScan) -> None:
        self._scan = scan

    # ------------------------------------------------------------------
    def step(self, now: float) -> List[TrackedObject]:
        if self._scan is None or not self._scan.ranges:
            return []

        clusters = self._cluster_points(self._scan)
        tracks = self._associate(clusters, now)
        self._prune_tracks(now)
        return tracks

    # ------------------------------------------------------------------
    def _cluster_points(self, scan: LaserScan) -> List[Dict[str, float]]:
        clusters: List[Dict[str, float]] = []
        current: List[Tuple[float, float, float]] = []  # (x, y, range)
        prev_point: Optional[Tuple[float, float]] = None

        angle = scan.angle_min
        for rng in scan.ranges:
            if math.isinf(rng) or math.isnan(rng):
                rng = scan.range_max
            if rng < scan.range_min or rng > scan.range_max:
                angle += scan.angle_increment
                prev_point = None
                if current:
                    self._finalise_cluster(current, clusters)
                    current = []
                continue

            x = rng * math.cos(angle)
            y = rng * math.sin(angle)
            point = (x, y)

            if prev_point is None:
                current = [(x, y, rng)]
            else:
                dist = math.hypot(point[0] - prev_point[0], point[1] - prev_point[1])
                if dist <= self.params.cluster_dist:
                    current.append((x, y, rng))
                else:
                    self._finalise_cluster(current, clusters)
                    current = [(x, y, rng)]

            prev_point = point
            angle += scan.angle_increment

        if current:
            self._finalise_cluster(current, clusters)

        return clusters

    def _finalise_cluster(self, cluster: List[Tuple[float, float, float]], clusters: List[Dict[str, float]]):
        if len(cluster) < self.params.min_points:
            return
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        rs = [p[2] for p in cluster]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        rng = sum(rs) / len(rs)
        theta = math.atan2(cy, cx)
        clusters.append({"x": cx, "y": cy, "range": rng, "theta": theta})

    # ------------------------------------------------------------------
    def _associate(self, clusters: List[Dict[str, float]], now: float) -> List[TrackedObject]:
        tracks: List[TrackedObject] = []
        used_tracks: Dict[int, bool] = {tid: False for tid in self._tracks}

        for cluster in clusters:
            best_id = None
            best_dist = float("inf")
            for tid, track in self._tracks.items():
                dt = max(1e-3, now - track.last_seen)
                pred_x = track.range * math.cos(track.theta) + track.vx * dt
                pred_y = track.range * math.sin(track.theta) + track.vy * dt
                dist = math.hypot(cluster["x"] - pred_x, cluster["y"] - pred_y)
                if dist < best_dist and dist <= self.params.match_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                tracked = self._update_track(best_id, cluster, now)
            elif len(self._tracks) + 1 <= self.params.max_tracks:
                tracked = self._create_track(cluster, now)
            else:
                continue

            used_tracks[tracked.id] = True
            tracks.append(tracked)

        # For tracks that were not matched this frame, keep them alive but mark.
        for tid, track in list(self._tracks.items()):
            if used_tracks.get(tid):
                continue
            if now - track.last_seen < self.params.vanish_time:
                tracks.append(track)
            else:
                self._tracks.pop(tid, None)

        return tracks

    def _create_track(self, cluster: Dict[str, float], now: float) -> TrackedObject:
        tid = self._next_id
        self._next_id += 1
        speed = 0.0
        track = TrackedObject(
            id=tid,
            range=cluster["range"],
            theta=cluster["theta"],
            vx=0.0,
            vy=0.0,
            speed=speed,
            is_dynamic=False,
            last_seen=now,
        )
        self._tracks[tid] = track
        return track

    def _update_track(self, tid: int, cluster: Dict[str, float], now: float) -> TrackedObject:
        track = self._tracks[tid]
        dt = max(1e-3, now - track.last_seen)

        alpha = max(0.0, min(1.0, self.params.smooth_alpha))
        prev_x = track.range * math.cos(track.theta)
        prev_y = track.range * math.sin(track.theta)
        pred_x = prev_x + track.vx * dt
        pred_y = prev_y + track.vy * dt

        new_x = alpha * cluster["x"] + (1.0 - alpha) * pred_x
        new_y = alpha * cluster["y"] + (1.0 - alpha) * pred_y

        vx = (new_x - prev_x) / dt
        vy = (new_y - prev_y) / dt
        speed = math.hypot(vx, vy)

        theta = math.atan2(new_y, new_x)
        rng = math.hypot(new_x, new_y)

        is_dynamic = speed > self.params.speed_thresh_moving

        updated = TrackedObject(
            id=tid,
            range=rng,
            theta=theta,
            vx=vx,
            vy=vy,
            speed=speed,
            is_dynamic=is_dynamic,
            last_seen=now,
        )
        self._tracks[tid] = updated
        return updated

    def _prune_tracks(self, now: float) -> None:
        to_remove = []
        for tid, track in self._tracks.items():
            if now - track.last_seen > self.params.max_age:
                to_remove.append(tid)
        for tid in to_remove:
            self._tracks.pop(tid, None)


__all__ = ["TrackParams", "TrackedObject", "DynamicTracker"]

