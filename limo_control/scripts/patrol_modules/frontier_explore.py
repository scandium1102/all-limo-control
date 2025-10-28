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
    def __init__(self, params: FrontierParams):
        self.params = params
        self._map: Optional[OccupancyGrid] = None
        self._map_array: Optional[np.ndarray] = None
        self._resolution: float = 0.05
        self._origin: Tuple[float, float] = (0.0, 0.0)
        self._robot_pose: Optional[Tuple[float, float, float]] = None
        self._last_goal: Optional[FrontierGoal] = None
        self._last_plan_time: float = 0.0

    def update_map(self, grid: OccupancyGrid) -> None:
        self._map = grid
        self._resolution = grid.info.resolution
        self._origin = (grid.info.origin.position.x, grid.info.origin.position.y)
        data = np.array(grid.data, dtype=int)
        self._map_array = data.reshape((grid.info.height, grid.info.width))

    def set_robot_pose(self, x: float, y: float, yaw: float) -> None:
        self._robot_pose = (x, y, yaw)

    def pick_next_goal(self) -> Optional[FrontierGoal]:
        now = time.time()
        if self._map_array is None or self._robot_pose is None:
            return None
        if now - self._last_plan_time < self.params.replan_period and self._last_goal is not None:
            if not self._goal_reached(self._last_goal):
                return self._last_goal
        self._last_plan_time = now

        frontier_mask = self._find_frontiers()
        clusters = self._cluster_frontiers(frontier_mask)
        if not clusters:
            self._last_goal = None
            return None

        best_goal: Optional[FrontierGoal] = None
        for cluster in clusters:
            if len(cluster) < self.params.min_cluster_size:
                continue
            goal = self._evaluate_cluster(cluster)
            if goal is None:
                continue
            if best_goal is None or goal.score > best_goal.score:
                best_goal = goal

        if best_goal is None:
            self._last_goal = None
            return None

        self._last_goal = best_goal
        return best_goal

    # ------------------------------------------------------------------
    def _find_frontiers(self) -> np.ndarray:
        assert self._map_array is not None
        free = self._map_array == 0
        unknown = self._map_array == -1
        frontier = np.zeros_like(self._map_array, dtype=bool)
        h, w = self._map_array.shape
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if not unknown[y, x]:
                    continue
                if np.any(free[y - 1 : y + 2, x - 1 : x + 2]):
                    frontier[y, x] = True
        return frontier

    def _cluster_frontiers(self, frontier_mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        visited = np.zeros_like(frontier_mask, dtype=bool)
        clusters: List[List[Tuple[int, int]]] = []
        h, w = frontier_mask.shape
        for y in range(h):
            for x in range(w):
                if not frontier_mask[y, x] or visited[y, x]:
                    continue
                cluster: List[Tuple[int, int]] = []
                queue = deque([(x, y)])
                visited[y, x] = True
                while queue:
                    cx, cy = queue.popleft()
                    cluster.append((cx, cy))
                    for nx in range(cx - 1, cx + 2):
                        for ny in range(cy - 1, cy + 2):
                            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                                continue
                            if visited[ny, nx] or not frontier_mask[ny, nx]:
                                continue
                            visited[ny, nx] = True
                            queue.append((nx, ny))
                clusters.append(cluster)
        return clusters

    def _evaluate_cluster(self, cluster: List[Tuple[int, int]]) -> Optional[FrontierGoal]:
        if self._robot_pose is None:
            return None
        robot_x, robot_y, _ = self._robot_pose
        centroid_x = sum(c[0] for c in cluster) / len(cluster)
        centroid_y = sum(c[1] for c in cluster) / len(cluster)
        world_x, world_y = self._map_to_world(centroid_x, centroid_y)
        if not self._is_reachable(world_x, world_y):
            return None
        distance = math.hypot(world_x - robot_x, world_y - robot_y)
        score = self.params.gain_weight * len(cluster) - self.params.dist_weight * distance
        heading = math.atan2(world_y - robot_y, world_x - robot_x)

        goal = FrontierGoal(map_xy=(world_x, world_y), heading_hint=heading, score=score)
        if self._last_goal is not None:
            last_x, last_y = self._last_goal.map_xy
            if math.hypot(world_x - last_x, world_y - last_y) < self.params.hysteresis_radius:
                score += 0.1
                goal = FrontierGoal(map_xy=(world_x, world_y), heading_hint=heading, score=score)
        return goal

    def _map_to_world(self, ix: float, iy: float) -> Tuple[float, float]:
        ox, oy = self._origin
        x = ox + (ix + 0.5) * self._resolution
        y = oy + (iy + 0.5) * self._resolution
        return x, y

    def _is_reachable(self, wx: float, wy: float) -> bool:
        if self._map_array is None:
            return False
        radius_cells = int(max(1, math.ceil(self.params.inflate_radius / self._resolution)))
        cx = int((wx - self._origin[0]) / self._resolution)
        cy = int((wy - self._origin[1]) / self._resolution)
        h, w = self._map_array.shape
        for y in range(max(0, cy - radius_cells), min(h, cy + radius_cells + 1)):
            for x in range(max(0, cx - radius_cells), min(w, cx + radius_cells + 1)):
                if math.hypot(x - cx, y - cy) > radius_cells:
                    continue
                val = self._map_array[y, x]
                if val > 70:
                    return False
        return True

    def _goal_reached(self, goal: FrontierGoal) -> bool:
        if self._robot_pose is None:
            return False
        robot_x, robot_y, _ = self._robot_pose
        return math.hypot(robot_x - goal.map_xy[0], robot_y - goal.map_xy[1]) <= self.params.goal_reached_tol
