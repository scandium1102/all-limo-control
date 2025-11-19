# LIMO Gazebo Refactor Log

## 0. Workspace Snapshot
- Workspace root: `~/limo_ws`
- Backed up legacy tree to `~/limo_ws_prev_20251117_171135` for reference.
- Current catkin packages under `~/limo_ws/src`: `limo_control`, `limo_ros`, `sim/*`, `ugv_gazebo_sim/*`.
- Extra assets: `aws-robomaker-hospital-world` (Gazebo world + models), `maps/`, `scripts/` (diagnostic + bringup).

## 1. Environment Preparation
1. Conda environment `ros_gazebo_38` is the canonical runtime; contains ROS Noetic-compatible Python (ros_numpy, etc.).
2. `~/.bashrc` now sources `/opt/ros/noetic/setup.bash` and `~/limo_ws/devel/setup.bash`, and exports Gazebo model/world paths (including AWS hospital world).
3. Added helper alias (manual step): run `conda activate ros_gazebo_38 && source ~/limo_ws/devel/setup.bash` before using ROS.
4. Fixed conflicting apt sources (VS Code) + refreshed ROS GPG key, then installed required packages:
   - `ros-noetic-pcl-ros`, `ros-noetic-depthimage-to-laserscan`, `ros-noetic-ros-numpy`, `ros-noetic-dynamic-reconfigure` and dependencies.

## 2. Repository Restructure
- Cloned official stacks (`limo_ros`, `ugv_gazebo_sim`) into `~/limo_ws/src`.
- Copied our maintained `limo_control` package out of the repo and rebuilt from scratch (this folder is now authoritative for catkin).
- Saved the original monorepo as `~/limo_ws/src/all_limo_control_repo` (contains the `.git` history). The repo now mirrors the catkin package content.
- Added `sim/` tree providing:
  - `limo_sim_description`: minimal URDF with LiDAR tilt + depth camera plugin.
  - `limo_sim_bringup`: world-loader, spawn, slam/nav launch files, world templates (`hospital_world.world`).
  - `sim/worlds/hospital_world.world` (placeholder) + AWS hospital world for large-scale testing.

## 3. Launch + Script Work
- `limo_control/launch/limo_sim_sensors.launch`: starts Gazebo empty world + spawn + diff drive controller + `/scan -> /limo/scan` relay + depth nodelets.
- Added `depth_cloud_pipeline.launch` (nodelets + depthimage_to_laserscan) for use with official launches.
- Added scenario launchers:
  - `depth_avoidance_sim.launch`
  - `lidar_avoidance_sim.launch`
- Created helper scripts in `~/limo_ws/scripts`:
  - `diag_ros_env.sh`: prints Conda/ROS/Gazebo paths/warnings.
  - `sim_bringup.sh`: clean residual Gazebo/ROS, run `catkin_make`, launch sim.
- Integrated AWS hospital world via `world:=/home/scandial/limo_ws/aws-robomaker-hospital-world/worlds/aws_hospital.world` args.

## 4. Issues & Fixes
| Issue | Symptom | Fix |
|------|---------|-----|
| Duplicate `roslaunch` targets | `limo_control` found twice | Moved git repo to `all_limo_control_repo`, kept only pure package in catkin src. |
| Depth nodelets missing | `pcl/CropBox` `depthimage_to_laserscan` not found | Installed `ros-noetic-pcl-ros` + `ros-noetic-depthimage-to-laserscan`. |
| `ros_numpy`/`limo_control.cfg` ImportError | depth/lidar avoidance crashed | Installed `ros-noetic-ros-numpy`; regenerated dynamic reconfigure modules and added import fallbacks. |
| Apt errors (`Signed-By`, ROS key expired) | `apt-get update` failed | Removed old VS Code list; re-imported ROS key, reran `apt-get update`. |
| Gazebo LiDAR topic mismatch | `/scan` only | Added `relay` to `/limo/scan`; pipeline generates `/camera/depth/points_filtered` + `/camera/depth/scan`. |
| Need large indoor world | Default world empty | Cloned `aws-robomaker-hospital-world`, updated Gazebo env vars. |

## 5. Dynamic Reconfigure Regeneration
- `CMakeLists.txt` now runs Python on `cfg/Avoidance.cfg` and `cfg/DepthHazard.cfg` via custom commands, creating both headers and Python modules in `devel`. Exported as `limo_control_gencfg` target.
- Modules installed to `<devel>/lib/python3/dist-packages/limo_control/cfg`; scripts dynamically import and fall back gracefully if missing.

## 6. Current Launch Recipe
1. **Gazebo (official diff-drive)**: `conda activate ros_gazebo_38 && roslaunch limo_gazebo_sim limo_four_diff.launch`
2. **Gazebo (custom)**: `roslaunch limo_control limo_sim_sensors.launch gui:=true world:=...aws_hospital.world`
3. **SLAM**: `roslaunch limo_control slam.launch enable_rviz:=false`
4. **Depth pipeline only**: `roslaunch limo_control depth_cloud_pipeline.launch scan_frame:=base_link`
5. **Depth avoidance**: `roslaunch limo_control depth_avoidance_sim.launch scan_topic:=/limo/scan depth_cloud_topic:=/camera/depth/points_filtered depth_scan_topic:=/camera/depth/scan`
6. **LiDAR avoidance**: `roslaunch limo_control lidar_avoidance_sim.launch scan_topic:=/limo/scan map_topic:=""`

TF warnings (`TF_REPEATED_DATA`) come from Gazebo when stationary and can be ignored. LiDAR timeout warnings indicate `/limo/scan` is not streaming—ensure TM1 or relay launch is active.

## 7. Pending / Notes
- When starting a fresh terminal: always `conda activate ros_gazebo_38 && source ~/limo_ws/devel/setup.bash` to ensure PYTHONPATH includes generated cfg modules.
- `roslaunch` commands should now succeed without manual workarounds.
- Dynamic reconfigure GUIs (rqt_reconfigure) can attach to `/depth_avoidance` or `/limo_lidar_avoidance` once those nodes are running.
