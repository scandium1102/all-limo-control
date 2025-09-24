#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frontier-based exploration helper."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rospy
from nav_msgs.msg import OccupancyGrid


@dataclass
class FrontierParams:
    min_cluster_size: int
    inflate_radius: float
    gain_weight: float
    dist_weight: float
    hysteresis_radius: float
    goal_reached_tol: float
    replan_period: float


@dataclass
class FrontierGoal:
    map_xy: Tuple[float, float]
    heading_hint: float
    score: float


class FrontierExplorer:
    """Simple frontier explorer working on OccupancyGrid maps."""

    def __init__(self, params: FrontierParams):
        self.params = params
        self._map: Optional[OccupancyGrid] = None
        self._map_array: Optional[List[int]] = None
        self._width = 0
        self._height = 0
        self._resolution = 0.05
        self._origin = (0.0, 0.0)
        self._robot_pose = (0.0, 0.0, 0.0)
        self._last_goal: Optional[FrontierGoal] = None
        self._last_plan_time = 0.0

    # ------------------------------------------------------------------
    def update_map(self, grid: OccupancyGrid) -> None:
        self._map = grid
        self._map_array = list(grid.data)
        self._width = grid.info.width
        self._height = grid.info.height
        self._resolution = grid.info.resolution
        self._origin = (grid.info.origin.position.x, grid.info.origin.position.y)

    def set_robot_pose(self, x: float, y: float, yaw: float) -> None:
        self._robot_pose = (x, y, yaw)

    # ------------------------------------------------------------------
    def pick_next_goal(self) -> Optional[FrontierGoal]:
        if self._map is None or self._map_array is None:
            return None

        now = self._now()
        if self._last_goal is not None:
            distance = self._distance_to_goal(self._last_goal.map_xy)
            if distance <= self.params.goal_reached_tol:
                self._last_goal = None
            elif now - self._last_plan_time < self.params.replan_period:
                return self._last_goal

        frontiers = self._extract_frontiers()
        if not frontiers:
            self._last_goal = None
            return None

        best_goal: Optional[FrontierGoal] = None
        best_score = -float("inf")

        for cluster in frontiers:
            score, goal = self._evaluate_cluster(cluster)
            if goal is None:
                continue
            if score > best_score:
                best_score = score
                best_goal = goal

        self._last_plan_time = now
        self._last_goal = best_goal
        return best_goal

    # ------------------------------------------------------------------
    def _extract_frontiers(self) -> List[List[Tuple[int, int]]]:
        if self._map_array is None:
            return []

        visited = [[False for _ in range(self._width)] for _ in range(self._height)]
        frontiers: List[List[Tuple[int, int]]] = []

        for r in range(self._height):
            for c in range(self._width):
                idx = self._index(r, c)
                value = self._map_array[idx]
                if value != -1 or visited[r][c]:
                    continue
                if not self._has_free_neighbor(r, c):
                    continue

                cluster: List[Tuple[int, int]] = []
                queue = [(r, c)]
                visited[r][c] = True

                while queue:
                    rr, cc = queue.pop(0)
                    cluster.append((rr, cc))
                    for nr, nc in self._neighbors(rr, cc):
                        if not self._in_bounds(nr, nc):
                            continue
                        if visited[nr][nc]:
                            continue
                        nidx = self._index(nr, nc)
                        if self._map_array[nidx] != -1:
                            continue
                        if not self._has_free_neighbor(nr, nc):
                            continue
                        visited[nr][nc] = True
                        queue.append((nr, nc))

                if len(cluster) >= self.params.min_cluster_size:
                    frontiers.append(cluster)

        return frontiers

    def _evaluate_cluster(self, cluster: List[Tuple[int, int]]) -> Tuple[float, Optional[FrontierGoal]]:
        world_points = [self._cell_to_world(r, c) for r, c in cluster]
        cx = sum(p[0] for p in world_points) / len(world_points)
        cy = sum(p[1] for p in world_points) / len(world_points)

        if not self._is_reachable(cx, cy):
            return -float("inf"), None

        robot_x, robot_y, _ = self._robot_pose
        dist = math.hypot(cx - robot_x, cy - robot_y)
        heading = math.atan2(cy - robot_y, cx - robot_x)

        score = self.params.gain_weight * len(cluster) - self.params.dist_weight * dist

        if self._last_goal is not None:
            prev_x, prev_y = self._last_goal.map_xy
            if math.hypot(cx - prev_x, cy - prev_y) < self.params.hysteresis_radius:
                score -= 0.5 * self.params.gain_weight * len(cluster)

        goal = FrontierGoal(map_xy=(cx, cy), heading_hint=heading, score=score)
        return score, goal

    # ------------------------------------------------------------------
    def _is_reachable(self, x: float, y: float) -> bool:
        radius = int(math.ceil(self.params.inflate_radius / max(1e-3, self._resolution)))
        cell = self._world_to_cell(x, y)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = cell[0] + dr
                cc = cell[1] + dc
                if not self._in_bounds(rr, cc):
                    return False
                idx = self._index(rr, cc)
                occ = self._map_array[idx]
                if occ == -1 or occ >= 50:
                    return False
        return True

    def _has_free_neighbor(self, r: int, c: int) -> bool:
        for nr, nc in self._neighbors(r, c):
            if not self._in_bounds(nr, nc):
                continue
            idx = self._index(nr, nc)
            if self._map_array[idx] == 0:
                return True
        return False

    def _neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        return [
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1),
            (r - 1, c - 1),
            (r - 1, c + 1),
            (r + 1, c - 1),
            (r + 1, c + 1),
        ]

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self._height and 0 <= c < self._width

    def _index(self, r: int, c: int) -> int:
        return r * self._width + c

    def _cell_to_world(self, r: int, c: int) -> Tuple[float, float]:
        x = self._origin[0] + (c + 0.5) * self._resolution
        y = self._origin[1] + (r + 0.5) * self._resolution
        return x, y

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        c = int((x - self._origin[0]) / self._resolution)
        r = int((y - self._origin[1]) / self._resolution)
        return r, c

    def _distance_to_goal(self, goal: Tuple[float, float]) -> float:
        x, y, _ = self._robot_pose
        return math.hypot(goal[0] - x, goal[1] - y)

    def _now(self) -> float:
        if rospy.is_shutdown():
            return time.time()
        try:
            return rospy.Time.now().to_sec()
        except rospy.ROSInitException:
            return time.time()


__all__ = ["FrontierParams", "FrontierGoal", "FrontierExplorer"]

