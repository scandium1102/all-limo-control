# patrol_modules consolidated package
"""
patrol_modules package (consolidated)

Exports common classes for quick import patterns:
  from patrol_modules import SensorHub, WallFollower, BackoffBehavior, StuckRecovery, ReturnHome, CoverageGrid
Also contains dynamic_tracker, frontier_explore, lidar_avoid modules for avoidance nodes.
"""

from .sensors import SensorHub
from .wall_follow import WallFollower
from .backoff import BackoffBehavior
from .stuck_recovery import StuckRecovery
from .return_home import ReturnHome
from .coverage_grid import CoverageGrid

__all__ = [
    "SensorHub",
    "WallFollower",
    "BackoffBehavior",
    "StuckRecovery",
    "ReturnHome",
    "CoverageGrid",
]
