# Architecture

## Boundary

This repository contains one Catkin package: `limo_control`. Earlier revisions
duplicated package files at the repository root and inside `limo_control/`.
The inner package contained the newer depth and exploration work, so it is now
the single source of truth.

The implementation is simulation-first. ROS nodes operate above the embedded
firmware layer; this repository does not contain LIMO controller firmware,
myCobot firmware, Jetson images, or physical calibration results.

## Runtime paths

### Modular patrol

```text
/limo/scan ─┐
/camera/depth/points ─┼─> SensorHub ─> WallFollower ─> /cmd_vel
/odom ──────┘                 │              │
                             │              └─> BackoffBehavior
                             ├─> StuckRecovery
                             ├─> CoverageGrid
                             └─> ReturnHome
```

`limo_patrol.py` coordinates these helpers with explicit `FWD`, `AVOID`,
`RECOVERY`, `RETURN`, and `STOP` states. When return-to-home avoidance runs,
the controller resumes `RETURN` rather than restarting normal patrol.

### LiDAR exploration and avoidance

```text
LaserScan ─> preprocessing ─> DynamicTracker ─┐
                                              ├─> LidarAvoider ─> cmd_vel
OccupancyGrid ─> FrontierExplorer ─> nav hint ┘
Odometry / TF ────────────────────────────────┘
```

`LidarAvoider` handles gap selection, heading smoothing, time-to-collision
limits, dynamic-obstacle inflation, acceleration limiting, and recovery.

### Depth-aware avoidance

```text
LaserScan + depth-derived scan + PointCloud2
                    │
                    ├─> synchronized LiDAR avoidance
                    └─> 3D plane/hazard analysis
                              │
                              ├─ slope
                              ├─ drop-off
                              └─ overhead clearance
```

The node falls back to LiDAR-only behavior when depth data times out.

### Global planning

`my_global_planner.cpp` implements `nav_core::BaseGlobalPlanner`. It converts
world poses to costmap cells, performs eight-connected A* search, reconstructs
the path, and exposes the plugin as
`my_planner_namespace/MyGlobalPlanner`.

## Source ownership

| Path | Responsibility |
| --- | --- |
| `limo_control/scripts/` | Executable ROS node entry points only |
| `limo_control/src/limo_control/patrol_modules/` | Reusable Python behavior and planning modules |
| `limo_control/src/my_global_planner.cpp` | Compiled C++ plugin |
| `limo_control/cfg/` | Dynamic-reconfigure parameter schemas |
| `limo_control/config/` | Runtime YAML configuration |
| `limo_control/launch/` | Process composition and topic remapping |
| `limo_control/examples/` | Historical reference code, not installed |

## Removed duplication

- the older root-level Catkin package mirror;
- the second `patrol_modules` mirror;
- committed `__pycache__` and `.pyc` files;
- obsolete `.bak` and `_backup.launch` files;
- `global_executor.py`, `map_saver.py`, and `sweep_planner.py`, which were six
  identical copies of the same patrol script and did not implement their names.

All removed content remains recoverable from Git history.
