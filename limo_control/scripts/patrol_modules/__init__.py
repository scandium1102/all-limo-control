"""Auxiliary modules for LiDAR-based avoidance and exploration."""

from .lidar_avoid import LidarAvoider, AvoidParams, GapInfo, DebugInfo
from .dynamic_tracker import DynamicTracker, TrackParams, TrackedObject
from .frontier_explore import FrontierExplorer, FrontierParams, FrontierGoal

__all__ = [
    "LidarAvoider",
    "AvoidParams",
    "GapInfo",
    "DebugInfo",
    "DynamicTracker",
    "TrackParams",
    "TrackedObject",
    "FrontierExplorer",
    "FrontierParams",
    "FrontierGoal",
]
