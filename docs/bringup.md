LIMO Bringup (Base + LiDAR + Depth)
===================================

Goals
-----
- One consistent LiDAR bringup for YDLIDAR Tmini Plus (USB serial).
- Avoid ROS overlay conflicts that cause checksum storms or missing topics.
- Optional depth camera pipeline with sane defaults.

Prerequisites
-------------
- Ubuntu 20.04 + ROS Noetic
- Packages: `ros-noetic-ydlidar-ros-driver`, `orbbec_camera` (if using depth)

Overlay Hygiene (very important)
--------------------------------
Problem: If your workspace contains another copy of `ydlidar_ros_driver`, sourcing `~/limo_ws/devel` makes ROS pick that copy instead of `/opt/ros/noetic`, often leading to checksum errors or topic differences.

Fix (choose one):
- Keep only the system package: move the workspace copy aside.
  
      mv ~/limo_ws/src/ydlidar_ros_driver ~/limo_ws/src/ydlidar_ros_driver.bak
      cd ~/limo_ws && catkin_make && source devel/setup.bash && rospack profile
      rospack find ydlidar_ros_driver  # should point into /opt/ros/noetic

- Or keep only the workspace copy: remove the system package.
  
      sudo apt remove ros-noetic-ydlidar-ros-driver
      cd ~/limo_ws && catkin_make && source devel/setup.bash && rospack profile

Helper: `scripts/check_overlay.sh` prints which driver is being used and detects duplicates.

LiDAR (USB, Tmini Plus)
-----------------------
Launch file: `limo_control/launch/sensors/lidar_tmini_usb.launch`

Parameters baked into the launch (validated for Tmini family):
- `port=/dev/ydlidar`, `baudrate=230400`
- `frame_id=laser_frame`, `lidar_type=1` (TOF)
- `resolution_fixed=true`, `intensity=true`, `frequency=8.0`, `auto_reconnect=true`

Static TF provided: `base_link -> laser_frame` (0,0,0,0,0,0; adjust later to match your mount).

Udev rule for stable `/dev/ydlidar`
------------------------------------
File: `udev/99-ydlidar.rules`

Install on the robot:

    sudo cp udev/99-ydlidar.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules && sudo udevadm trigger
    # replug LiDAR, verify: ls -l /dev/ydlidar

Bringup Options
---------------
One-shot bringup (base + LiDAR):

    roslaunch limo_control bringup/minimal_bringup.launch

Three-step bringup:

    roscore
    roslaunch limo_base limo_base.launch
    roslaunch limo_control sensors/lidar_tmini_usb.launch

Checks (acceptance):
- No continuous "Check Sum" errors within 30s
- `rostopic hz /scan` ≈ 8 Hz
- `rosrun tf tf_echo base_link laser_frame` outputs continuously

Depth Camera (optional)
-----------------------
For Gemini 330 series with v4l2 backend:

    roslaunch limo_control sensors/orbbec_gemini330_v4l2.launch enable_depth:=true enable_color:=true device_num:=0

Then launch the depth pipeline (defaults to `depth/image_raw`; change via `image_topic`):

    roslaunch limo_control depth_cloud_pipeline.launch scan_frame:=base_link image_topic:=depth/image_raw

Twist mux coordination
----------------------
`avoid_only.launch` brings up the only `twist_mux`. `avoid/avoid_with_depth.launch` has `start_mux:=false` by default to avoid duplicates. If you run only depth avoidance, pass `start_mux:=true`.

APT: Husarnet key
------------------
If `apt update` shows `EXPKEYSIG 197D62F68A4C7BD6`, update or disable the repo:

    sudo ./scripts/fix_husarnet_apt_key.sh

Notes
-----
- Always `source /opt/ros/noetic/setup.bash` then `source ~/limo_ws/devel/setup.bash` in new terminals.
- If LiDAR still prints checksum errors, re-check overlay (no duplicate drivers), cable quality, and port/baud in the launch.

