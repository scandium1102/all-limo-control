"""Frontier based exploration helper."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
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
    """Detect and score frontier clusters from occupancy grid maps."""

    def __init__(self, params: FrontierParams):
        self.params = params
        self._map: Optional[OccupancyGrid] = None
        self._map_array: Optional[np.ndarray] = None
        self._robot_pose: Optional[Tuple[float, float, float]] = None
        self._last_goal: Optional[FrontierGoal] = None
        self._last_plan_time: float = 0.0

    # ------------------------------------------------------------------
    def update_map(self, grid: OccupancyGrid) -> None:
        self._map = grid
        data = np.array(grid.data, dtype=np.int8)
        self._map_array = data.reshape((grid.info.height, grid.info.width))

    # ------------------------------------------------------------------
    def set_robot_pose(self, x: float, y: float, yaw: float) -> None:
        self._robot_pose = (x, y, yaw)

    # ------------------------------------------------------------------
    def pick_next_goal(self) -> Optional[FrontierGoal]:
        if self._map is None or self._map_array is None:
            return None
        if self._robot_pose is None:
            return None

        now = self._time()
        if self._last_goal and self._distance_to_goal(self._last_goal.map_xy) <= self.params.goal_reached_tol:
            self._last_goal = None

        if self._last_goal and now - self._last_plan_time < self.params.replan_period:
            return self._last_goal

        frontiers = self._detect_frontiers()
        if not frontiers:
            self._last_goal = None
            return None

        scored = [self._score_frontier(cluster) for cluster in frontiers]
        scored = [goal for goal in scored if goal is not None]
        if not scored:
            self._last_goal = None
            return None

        scored.sort(key=lambda g: g.score, reverse=True)
        best = scored[0]

        if self._last_goal:
            dist = self._distance(best.map_xy, self._last_goal.map_xy)
            if dist <= self.params.hysteresis_radius:
                best = self._last_goal

        self._last_goal = best
        self._last_plan_time = now
        return best

    # ------------------------------------------------------------------
    def _detect_frontiers(self) -> List[List[Tuple[int, int]]]:
        assert self._map is not None
        assert self._map_array is not None
        grid = self._map_array
        height, width = grid.shape
        frontier_cells: List[Tuple[int, int]] = []
        free_threshold = 30
        for y in range(height):
            for x in range(width):
                occ = grid[y, x]
                if occ != -1:
                    continue
                if self._has_free_neighbor(grid, x, y, free_threshold):
                    frontier_cells.append((x, y))

        visited = set()
        clusters: List[List[Tuple[int, int]]] = []
        for cell in frontier_cells:
            if cell in visited:
                continue
            queue = deque([cell])
            cluster: List[Tuple[int, int]] = []
            while queue:
                cx, cy = queue.popleft()
                if (cx, cy) in visited:
                    continue
                visited.add((cx, cy))
                cluster.append((cx, cy))
                for nx in range(cx - 1, cx + 2):
                    for ny in range(cy - 1, cy + 2):
                        if (nx, ny) in visited:
                            continue
                        if (nx, ny) in frontier_cells:
                            queue.append((nx, ny))
            if len(cluster) >= self.params.min_cluster_size:
                clusters.append(cluster)
        return clusters

    # ------------------------------------------------------------------
    def _has_free_neighbor(self, grid: np.ndarray, x: int, y: int, threshold: int) -> bool:
        height, width = grid.shape
        for nx in range(max(0, x - 1), min(width - 1, x + 1) + 1):
            for ny in range(max(0, y - 1), min(height - 1, y + 1) + 1):
                if nx == x and ny == y:
                    continue
                if grid[ny, nx] >= 0 and grid[ny, nx] <= threshold:
                    return True
        return False

    # ------------------------------------------------------------------
    def _score_frontier(self, cluster: List[Tuple[int, int]]) -> Optional[FrontierGoal]:
        assert self._map is not None
        assert self._map_array is not None
        if not cluster:
            return None

        center_px = np.mean([c[0] for c in cluster])
        center_py = np.mean([c[1] for c in cluster])
        world_x, world_y = self._pixel_to_world(center_px, center_py)

        if not self._is_accessible(center_px, center_py):
            return None

        robot_x, robot_y, _ = self._robot_pose  # type: ignore
        dist = self._distance((robot_x, robot_y), (world_x, world_y))
        score = self.params.gain_weight * len(cluster) - self.params.dist_weight * dist
        heading = math.atan2(world_y - robot_y, world_x - robot_x)
        return FrontierGoal(map_xy=(world_x, world_y), heading_hint=heading, score=score)

    # ------------------------------------------------------------------
    def _is_accessible(self, px: float, py: float) -> bool:
        assert self._map is not None
        assert self._map_array is not None
        resolution = self._map.info.resolution
        radius_cells = max(1, int(round(self.params.inflate_radius / max(resolution, 1e-3))))
        cx = int(round(px))
        cy = int(round(py))
        height, width = self._map_array.shape
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    return False
                if self._map_array[ny, nx] > 50:
                    return False
        return True

    # ------------------------------------------------------------------
    def _pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        assert self._map is not None
        origin = self._map.info.origin.position
        resolution = self._map.info.resolution
        x = origin.x + (px + 0.5) * resolution
        y = origin.y + (py + 0.5) * resolution
        return x, y

    # ------------------------------------------------------------------
    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # ------------------------------------------------------------------
    def _distance_to_goal(self, goal_xy: Tuple[float, float]) -> float:
        if self._robot_pose is None:
            return float("inf")
        robot_x, robot_y, _ = self._robot_pose
        return self._distance((robot_x, robot_y), goal_xy)

    # ------------------------------------------------------------------
    @staticmethod
    def _time() -> float:
        try:
            import rospy

            return rospy.get_time()
        except Exception:
            return time.time()

