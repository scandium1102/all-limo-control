# LIMO Control

ROS Noetic navigation and reactive-avoidance experiments for the AgileX LIMO
platform. The repository records a student simulation prototype; it does not
claim physical-robot validation, hospital deployment, or safety certification.

## What is included

- a C++ A* global-planner plugin for `move_base`;
- Python nodes for LiDAR avoidance, depth-aware hazard handling, and a modular
  patrol/return controller;
- reusable Python modules for sensor access, wall following, recovery,
  frontier selection, and dynamic-object tracking;
- ROS launch files, costmap settings, and dynamic-reconfigure definitions.

The detailed data flow and ownership of each folder are documented in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Repository layout

```text
.
├── .github/workflows/       Static repository checks
├── docs/                    Architecture and maintenance notes
├── limo_control/            The only ROS package in this repository
│   ├── cfg/                 dynamic_reconfigure schemas
│   ├── config/              Navigation and avoidance parameters
│   ├── examples/            Historical all-in-one prototype
│   ├── launch/              Simulation, sensor, SLAM, and avoidance launchers
│   ├── scripts/             Executable ROS nodes
│   └── src/
│       ├── limo_control/     Importable Python package
│       └── my_global_planner.cpp
└── tools/check_repository.py
```

## Main nodes

| Node | Purpose | Main output |
| --- | --- | --- |
| `limo_patrol.py` | Wall-follow patrol, backoff, recovery, coverage tracking, and return | `/cmd_vel` |
| `lidar_avoidance_node.py` | Follow-the-gap avoidance, frontier hints, and dynamic tracking | configurable, default `/cmd_vel` |
| `depth_avoidance_node.py` | LiDAR/depth fusion with slope, drop, and overhead hazard checks | configurable, default `/cmd_vel` |
| `my_global_planner.cpp` | A* global path generation for `move_base` | `nav_core::BaseGlobalPlanner` plugin |

`limo_control/examples/limo_patrol_v6.py` is retained as historical reference
and is not installed as a runtime node.

## Build in a ROS Noetic workspace

```bash
source /opt/ros/noetic/setup.bash
mkdir -p ~/limo_ws/src
cd ~/limo_ws/src
git clone https://github.com/scandium1102/all-limo-control.git
cd ~/limo_ws
catkin_make
source devel/setup.bash
```

The launch files also expect the relevant LIMO simulation, sensor, and
navigation packages listed in `limo_control/package.xml`. Hardware-specific
topic names, transforms, and sensor calibration must be verified before use.

Example launch commands:

```bash
roslaunch limo_control limo_sim_sensors.launch
roslaunch limo_control avoid_only.launch
```

For the physical-sensor launch path:

```bash
roslaunch limo_control avoid/avoid_with_depth.launch
```

## Validation

The cross-platform static check parses every Python and XML/launch file,
rejects generated/backup artifacts, verifies the single-package layout, and
detects exact duplicate files:

```bash
python3 tools/check_repository.py
```

A real `catkin_make` and Gazebo runtime test still require Ubuntu 20.04 with
ROS Noetic and the declared ROS dependencies.

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
