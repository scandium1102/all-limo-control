#!/usr/bin/env bash
set -euo pipefail

if ! command -v rosbag >/dev/null 2>&1; then
  echo "rosbag executable not found" >&2
  exit 1
fi

rosrun topic_tools throttle messages /camera/depth/points 10 /camera/depth/points_throttled &
THROTTLE_PID=$!
trap 'kill $THROTTLE_PID >/dev/null 2>&1 || true' EXIT

rosbag record -O avoid_debug.bag \
  /scan \
  /camera/depth/scan \
  /camera/depth/points_throttled \
  /cmd_vel \
  /limo_lidar_avoidance/avoidance_debug \
  /avoidance_debug_3d \
  /tf
