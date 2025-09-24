"""Frontier exploration helper for autonomous coverage."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid


@dataclass
class FrontierParams:
    min_cluster_size: int
    inflate_radius: float         # m，地圖通行膨脹
    gain_weight: float            # 收益評分權重
    dist_weight: float            # 距離懲罰權重
    hysteresis_radius: float      # m，避免前後抖動
    goal_reached_tol: float       # m
    replan_period: float          # s


@dataclass
class FrontierGoal:
    map_xy: Tuple[float, float]   # 地圖座標
    heading_hint: float           # rad，給避障的方向偏好
    score: float


class FrontierExplorer:
    """Extract frontiers from an occupancy grid and choose the next exploration goal."""

    def __init__(self, params: FrontierParams):
        self.params = params
        self._map: Optional[np.ndarray] = None
        self._origin = (0.0, 0.0)
        self._resolution = 0.05
        self._width = 0
        self._height = 0
        self._robot_pose = (0.0, 0.0, 0.0)
        self._last_goal: Optional[FrontierGoal] = None
        self._last_goal_time = 0.0

    # ------------------------------------------------------------------
    def update_map(self, grid: OccupancyGrid) -> None:
        data = np.array(grid.data, dtype=np.int16).reshape((grid.info.height, grid.info.width))
        self._map = data
        self._origin = (grid.info.origin.position.x, grid.info.origin.position.y)
        self._resolution = grid.info.resolution
        self._width = grid.info.width
        self._height = grid.info.height

    # ------------------------------------------------------------------
    def set_robot_pose(self, x: float, y: float, yaw: float) -> None:
        self._robot_pose = (x, y, yaw)

    # ------------------------------------------------------------------
    def pick_next_goal(self) -> Optional[FrontierGoal]:
        if self._map is None:
            return None

        now = rospy.get_time()
        robot_x, robot_y, _ = self._robot_pose

        if self._last_goal is not None:
            goal_x, goal_y = self._last_goal.map_xy
            dist = math.hypot(goal_x - robot_x, goal_y - robot_y)
            if dist <= self.params.goal_reached_tol:
                self._last_goal = None
            elif now - self._last_goal_time < self.params.replan_period:
                return self._last_goal

        free_mask = np.logical_and(self._map >= 0, self._map < 50)
        unknown_mask = self._map == -1

        frontier_mask = self._compute_frontier_mask(free_mask, unknown_mask)
        clusters = self._extract_clusters(frontier_mask)
        if not clusters:
            self._last_goal = None
            return None

        best_goal: Optional[FrontierGoal] = None
        best_score = -float("inf")
        last_goal_xy = self._last_goal.map_xy if self._last_goal else None

        for cluster in clusters:
            if len(cluster) < self.params.min_cluster_size:
                continue
            centroid_row = sum(pt[0] for pt in cluster) / len(cluster)
            centroid_col = sum(pt[1] for pt in cluster) / len(cluster)
            world_x, world_y = self._cell_to_world(centroid_row, centroid_col)
            if not self._reachable(world_x, world_y, free_mask):
                continue

            dist = math.hypot(world_x - robot_x, world_y - robot_y)
            score = self.params.gain_weight * len(cluster) - self.params.dist_weight * dist
            if last_goal_xy is not None:
                if math.hypot(world_x - last_goal_xy[0], world_y - last_goal_xy[1]) < self.params.hysteresis_radius:
                    score += 0.5 * self.params.gain_weight * len(cluster)

            if score > best_score:
                heading = math.atan2(world_y - robot_y, world_x - robot_x)
                best_score = score
                best_goal = FrontierGoal(map_xy=(world_x, world_y), heading_hint=heading, score=score)

        if best_goal is not None:
            self._last_goal = best_goal
            self._last_goal_time = now
        else:
            self._last_goal = None
        return best_goal

    # ------------------------------------------------------------------
    def _compute_frontier_mask(self, free_mask: np.ndarray, unknown_mask: np.ndarray) -> np.ndarray:
        frontier = np.zeros_like(free_mask, dtype=bool)
        padded_unknown = np.pad(unknown_mask, 1, mode="constant", constant_values=False)

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted_unknown = padded_unknown[1 + dr : 1 + dr + self._height, 1 + dc : 1 + dc + self._width]
                frontier |= free_mask & shifted_unknown

        # Mask out cells that are not free (safety)
        frontier &= free_mask
        return frontier

    # ------------------------------------------------------------------
    def _extract_clusters(self, frontier_mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        visited = np.zeros_like(frontier_mask, dtype=bool)
        clusters: List[List[Tuple[int, int]]] = []

        for r in range(self._height):
            for c in range(self._width):
                if not frontier_mask[r, c] or visited[r, c]:
                    continue
                cluster = []
                q = deque([(r, c)])
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    cluster.append((cr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < self._height and 0 <= nc < self._width:
                                if frontier_mask[nr, nc] and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    q.append((nr, nc))
                clusters.append(cluster)
        return clusters

    # ------------------------------------------------------------------
    def _cell_to_world(self, row: float, col: float) -> Tuple[float, float]:
        x = self._origin[0] + (col + 0.5) * self._resolution
        y = self._origin[1] + (row + 0.5) * self._resolution
        return x, y

    # ------------------------------------------------------------------
    def _reachable(self, world_x: float, world_y: float, free_mask: np.ndarray) -> bool:
        radius = max(0.0, self.params.inflate_radius)
        if radius <= 0.0:
            return True

        col = int(round((world_x - self._origin[0]) / self._resolution - 0.5))
        row = int(round((world_y - self._origin[1]) / self._resolution - 0.5))
        rad_cells = int(math.ceil(radius / self._resolution))
        for rr in range(row - rad_cells, row + rad_cells + 1):
            for cc in range(col - rad_cells, col + rad_cells + 1):
                if 0 <= rr < self._height and 0 <= cc < self._width:
                    dist = math.hypot((rr - row) * self._resolution, (cc - col) * self._resolution)
                    if dist <= radius and not free_mask[rr, cc]:
                        return False
                else:
                    return False
        return True

