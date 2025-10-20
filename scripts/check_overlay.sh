#!/usr/bin/env bash
set -euo pipefail

echo "==> Checking ROS overlay and ydlidar driver source"

if ! command -v rospack >/dev/null 2>&1; then
  echo "rospack not found. Please: source /opt/ros/noetic/setup.bash" >&2
  exit 2
fi

if [ -z "${ROS_PACKAGE_PATH:-}" ]; then
  echo "ROS_PACKAGE_PATH is empty; did you source your workspace?" >&2
fi

echo "ROS_PACKAGE_PATH="
printf '%s\n' "$ROS_PACKAGE_PATH" | tr ':' '\n'

echo
echo "rospack find ydlidar_ros_driver:"
yd_path=$(rospack find ydlidar_ros_driver 2>/dev/null || true)
if [ -n "$yd_path" ]; then
  echo " -> $yd_path"
else
  echo " -> NOT FOUND (install with: sudo apt install ros-noetic-ydlidar-ros-driver)"
fi

echo
echo "Searching for duplicate ydlidar_ros_driver folders in ROS_PACKAGE_PATH..."
dups=()
IFS=':' read -r -a paths <<<"${ROS_PACKAGE_PATH:-}"
for p in "${paths[@]}"; do
  [ -d "$p" ] || continue
  while IFS= read -r -d '' d; do
    dups+=("$d")
  done < <(find "$p" -maxdepth 4 -type d -name ydlidar_ros_driver -print0 2>/dev/null || true)
done

if [ ${#dups[@]} -gt 1 ]; then
  echo "Found multiple ydlidar_ros_driver directories:"
  printf ' - %s\n' "${dups[@]}"
  echo "Recommend keeping only one (prefer /opt/ros/noetic) and moving the others aside."
else
  echo "OK: No duplicates found (or only one)."
fi

echo
echo "Check recommended parameters from running node (if active):"
if rosparam get /ydlidar_node >/dev/null 2>&1; then
  rosparam get /ydlidar_node/lidar_type || true
  rosparam get /ydlidar_node/resolution_fixed || true
  rosparam get /ydlidar_node/intensity || true
else
  echo "ydlidar_node not running. Launch it, then re-run this script."
fi

