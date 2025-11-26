#!/usr/bin/env python3
"""
room_segmenter.py

Segment an occupancy grid into room-like regions by free-space connectivity with
minimum clearance checks (to cut narrow doorways), then publish centroids and
markers and optionally save to YAML.
"""

from __future__ import annotations

import math
import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class RoomRegion:
    name: str
    cells: List[Tuple[int, int]]  # (y, x) indices
    area_m2: float
    centroid: Tuple[float, float]
    bbox_min: Tuple[float, float]
    bbox_max: Tuple[float, float]
    min_clearance: float


class RoomSegmenter:
    def __init__(self):
        self.map_topic = rospy.get_param("~map_topic", "/map")
        self.min_clearance_m = float(rospy.get_param("~min_clearance_m", 0.35))
        self.min_room_area = float(rospy.get_param("~min_room_area", 2.0))
        self.unknown_as_obstacle = bool(rospy.get_param("~unknown_as_obstacle", True))
        self.output_yaml = rospy.get_param("~output_yaml", "")
        self.marker_ns = rospy.get_param("~marker_ns", "rooms")
        self.marker_lifetime = float(rospy.get_param("~marker_lifetime", 0.0))

        self._lock = threading.RLock()
        self._last_regions: List[RoomRegion] = []
        self._last_map: Optional[OccupancyGrid] = None

        self._pose_pub = rospy.Publisher("rooms/centers", PoseArray, queue_size=1, latch=True)
        self._marker_pub = rospy.Publisher("rooms/markers", MarkerArray, queue_size=1, latch=True)

        self._save_srv = rospy.Service("~save", Trigger, self._on_save)

        rospy.Subscriber(self.map_topic, OccupancyGrid, self._map_cb, queue_size=1)

    # ------------------------------------------------------------------ utils
    def _map_cb(self, msg: OccupancyGrid):
        with self._lock:
            self._last_map = msg
        regions = self._segment(msg)
        with self._lock:
            self._last_regions = regions
        self._publish(regions, msg)
        if self.output_yaml:
            self._save_yaml(self.output_yaml, regions, msg, quiet=True)

    def _segment(self, grid: OccupancyGrid) -> List[RoomRegion]:
        res = grid.info.resolution
        w = grid.info.width
        h = grid.info.height
        if w == 0 or h == 0 or res <= 0.0:
            return []

        data = np.array(grid.data, dtype=np.int16).reshape((h, w))
        free_mask = data == 0
        obstacle_mask = data > 50
        if self.unknown_as_obstacle:
            obstacle_mask |= data == -1

        # Multi-source distance transform (Manhattan) from obstacles
        clearance_cells = self._distance_to_obstacle(obstacle_mask)
        pass_thresh_cells = max(1, int(math.ceil(self.min_clearance_m / res)))

        visited = np.zeros_like(free_mask, dtype=bool)
        regions: List[RoomRegion] = []
        name_id = 1
        for y in range(h):
            for x in range(w):
                if visited[y, x] or not free_mask[y, x]:
                    continue
                if clearance_cells[y, x] < pass_thresh_cells:
                    visited[y, x] = True
                    continue
                cells = self._grow_component((y, x), free_mask, clearance_cells, visited, pass_thresh_cells)
                if not cells:
                    continue
                area = len(cells) * res * res
                if area < self.min_room_area:
                    continue
                cy, cx = np.mean(cells, axis=0)
                centroid = self._cell_to_world(int(cx), int(cy), grid)
                ys = [c[0] for c in cells]
                xs = [c[1] for c in cells]
                min_w = self._cell_to_world(min(xs), min(ys), grid)
                max_w = self._cell_to_world(max(xs), max(ys), grid)
                min_clear = float(np.min(clearance_cells[[c[0] for c in cells], [c[1] for c in cells]]) * res)
                regions.append(
                    RoomRegion(
                        name=f"room_{name_id}",
                        cells=cells,
                        area_m2=area,
                        centroid=centroid,
                        bbox_min=min_w,
                        bbox_max=max_w,
                        min_clearance=min_clear,
                    )
                )
                name_id += 1
        return regions

    def _distance_to_obstacle(self, obstacle_mask: np.ndarray) -> np.ndarray:
        h, w = obstacle_mask.shape
        inf = h * w + 1
        dist = np.full((h, w), inf, dtype=np.int32)
        q: deque = deque()
        obs_y, obs_x = np.nonzero(obstacle_mask)
        for y, x in zip(obs_y, obs_x):
            dist[y, x] = 0
            q.append((y, x))

        while q:
            y, x = q.popleft()
            d = dist[y, x] + 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if d < dist[ny, nx]:
                    dist[ny, nx] = d
                    q.append((ny, nx))
        return dist

    def _grow_component(
        self,
        start: Tuple[int, int],
        free: np.ndarray,
        clearance: np.ndarray,
        visited: np.ndarray,
        thresh: int,
    ) -> List[Tuple[int, int]]:
        h, w = free.shape
        comp: List[Tuple[int, int]] = []
        q: deque = deque([start])
        visited[start[0], start[1]] = True
        while q:
            y, x = q.popleft()
            comp.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or not free[ny, nx]:
                    continue
                if clearance[ny, nx] < thresh or min(clearance[ny, nx], clearance[y, x]) < thresh:
                    visited[ny, nx] = True
                    continue
                visited[ny, nx] = True
                q.append((ny, nx))
        return comp

    def _cell_to_world(self, cx: int, cy: int, grid: OccupancyGrid) -> Tuple[float, float]:
        res = grid.info.resolution
        ox = grid.info.origin.position.x
        oy = grid.info.origin.position.y
        wx = ox + (cx + 0.5) * res
        wy = oy + (cy + 0.5) * res
        return wx, wy

    # ------------------------------------------------------------------ publish/save
    def _publish(self, regions: List[RoomRegion], grid: OccupancyGrid):
        pa = PoseArray()
        pa.header.frame_id = grid.header.frame_id or "map"
        pa.header.stamp = rospy.Time.now()
        for r in regions:
            ps = PoseStamped()
            ps.header = pa.header
            ps.pose.position.x = r.centroid[0]
            ps.pose.position.y = r.centroid[1]
            ps.pose.orientation.w = 1.0
            pa.poses.append(ps.pose)
        self._pose_pub.publish(pa)

        markers = MarkerArray()
        for idx, r in enumerate(regions, start=1):
            m = Marker()
            m.header.frame_id = pa.header.frame_id
            m.header.stamp = pa.header.stamp
            m.ns = self.marker_ns
            m.id = idx
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x = r.centroid[0]
            m.pose.position.y = r.centroid[1]
            m.pose.position.z = 0.4
            m.pose.orientation.w = 1.0
            m.text = f"{r.name}\n{r.area_m2:.1f} m^2"
            m.scale.z = 0.25
            m.color.r = 0.1
            m.color.g = 0.6
            m.color.b = 0.9
            m.color.a = 1.0
            m.lifetime = rospy.Duration(self.marker_lifetime)
            markers.markers.append(m)
        self._marker_pub.publish(markers)

    def _save_yaml(self, path: str, regions: List[RoomRegion], grid: OccupancyGrid, quiet: bool = False) -> bool:
        try:
            import yaml
        except ImportError:
            if not quiet:
                rospy.logerr("PyYAML not available; cannot save YAML")
            return False

        data = []
        for r in regions:
            data.append(
                {
                    "name": r.name,
                    "area_m2": r.area_m2,
                    "centroid": {"x": r.centroid[0], "y": r.centroid[1]},
                    "bbox_min": {"x": r.bbox_min[0], "y": r.bbox_min[1]},
                    "bbox_max": {"x": r.bbox_max[0], "y": r.bbox_max[1]},
                    "min_clearance_m": r.min_clearance,
                }
            )

        payload = {
            "frame_id": grid.header.frame_id or "map",
            "resolution": grid.info.resolution,
            "origin": {
                "x": grid.info.origin.position.x,
                "y": grid.info.origin.position.y,
                "z": grid.info.origin.position.z,
            },
            "rooms": data,
        }

        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, sort_keys=False)
            if not quiet:
                rospy.loginfo("Saved room regions to %s", path)
            return True
        except OSError as exc:
            if not quiet:
                rospy.logerr("Failed to save YAML to %s: %s", path, exc)
            return False

    def _on_save(self, _req):
        with self._lock:
            regions = list(self._last_regions)
            grid = self._last_map
        resp = TriggerResponse()
        if grid is None or not regions:
            resp.success = False
            resp.message = "No map or regions available"
            return resp
        path = self.output_yaml or "/tmp/room_segments.yaml"
        ok = self._save_yaml(path, regions, grid, quiet=False)
        resp.success = ok
        resp.message = f"Saved to {path}" if ok else "Save failed"
        return resp


def main():
    rospy.init_node("room_segmenter")
    RoomSegmenter()
    rospy.loginfo("room_segmenter started")
    rospy.spin()


if __name__ == "__main__":
    main()
