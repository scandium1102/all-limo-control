# Session Log

## What we changed
- Switched move_base global planner to the official `global_planner/GlobalPlanner` (no custom MyGlobalPlanner in use).
- Integrated explore_lite + move_base + safety stack:
  - Safety filter `cmd_vel_safety_filter.py` (distance/TTC stop/slow).
  - Arbiter `cmd_vel_arbiter.py` (fallback to `lidar_avoidance_node` on move_base failure/timeout).
  - `lidar_avoidance_node.py` path guard added; patrol_modules relocated into the package root to ensure imports.
  - `explore_supervisor.py` (coverage milestone markers + auto home goal on completion).
- Added NLU/room pipeline:
  - `gemini_nlu.py` (API key via param/env, no hardcoded key).
  - `nlu_room_bridge.py` (text → NLU → GoToRoom).
  - `room_navigator.py` / `room_segmenter.py` for room goals.
- Web commander updates:
  - `app.py` uses command topic (`/vln_command`) instead of calling Gemini directly.
  - Saves locations to `locations.json` and publishes `/rooms/markers` so room_navigator can ingest.
  - package.xml/CMakeLists updated with missing deps and launch install.
- Cleaned workspace duplication: legacy copies moved to `limo_control_legacy_20250324` and `legacy_root_20250324` with CATKIN_IGNORE; active package is `src/all_limo_control_repo/limo_control`.

## Issues encountered & fixes
- `ModuleNotFoundError: patrol_modules.*`: fixed by moving patrol_modules into package root, enabling `catkin_python_setup()`, and adding a path guard in `lidar_avoidance_node.py`. Always source `/opt/ros/noetic/setup.bash` + `~/limo_ws/devel/setup.bash` before running.
- TF/costmap “Waiting for map/odom” / LiDAR timeout: occurs if Gazebo/SLAM not started or scan topic mismatched. Ensure Gazebo + SLAM are running and scan_topic is set to `/limo/scan`.
- TF_REPEATED_DATA warnings: from Gazebo publishing odom/base_footprint timestamps repeatedly; usually harmless, but ensure only one odom publisher.
- Legacy launch/duplicate packages: legacy folders now ignored; only `all_limo_control_repo/limo_control` is active.

## Current launch order (Gazebo simulation)
1. Common env (each terminal):
   ```
   source /opt/ros/noetic/setup.bash
   source ~/limo_ws/devel/setup.bash
   export PYTHONPATH=~/limo_ws/devel/lib/python3/dist-packages:$PYTHONPATH
   ```
2. TM1 Gazebo (four-wheel diff):
   ```
   roslaunch limo_gazebo_sim limo_four_diff.launch
   ```
3. TM2 SLAM (map → odom):
   ```
   roslaunch limo_control slam.launch use_sim_time:=true
   ```
4. TM3 Exploration/Nav/Safety:
   ```
   roslaunch limo_control explore_lite.launch scan_topic:=/limo/scan
   ```
   (explore_lite + move_base + safety + arbiter + lidar_avoid + supervisor)
5. Optional NLU/rooms:
   ```
   roslaunch limo_control gemini_nlu.launch api_key:=<KEY> command_topic:=/vln_command
   ```
6. Optional Web UI:
   ```
   roslaunch limo_web_commander web_commander.launch api_key:=<KEY>
   # browse http://0.0.0.0:5000
   ```

## How to observe what the robot is doing
- Avoidance debug: `rostopic echo /limo_lidar_avoidance/avoidance_debug` (state, notes: proximity_stop/ttc_stop/stuck_detected, d_min, ttc_min).
- Final cmd_vel: `rostopic echo /cmd_vel`.
- Navigation status/goals: `rostopic echo /move_base/status`, `/move_base_simple/goal`.
- Map/frontiers: RViz (map, costmap, markers); check `/rooms/markers` for room labels; coverage markers from supervisor.

## Remaining to test/verify
- Confirm patrol_modules import is stable across new terminals (with env + PYTHONPATH set).
- Verify scan topic `/limo/scan` present before launching explore_lite; ensure SLAM provides map→odom TF.
- LLM pipeline end-to-end: `/vln_command` → goal → move_base motion.
