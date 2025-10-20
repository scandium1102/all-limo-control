#!/usr/bin/env bash
set -euo pipefail

# Source ROS and workspace
[ -f /opt/ros/noetic/setup.bash ] && source /opt/ros/noetic/setup.bash
[ -f "$HOME/limo_ws/devel/setup.bash" ] && source "$HOME/limo_ws/devel/setup.bash"

# Set ROS master locally (derive IP)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/set_master_local.sh"

# Optional components via env vars (1/0)
WITH_DEPTH=${WITH_DEPTH:-0}
WITH_AVOID=${WITH_AVOID:-1}
WITH_SLAM=${WITH_SLAM:-1}
WITH_RVIZ=${WITH_RVIZ:-0}
SESSION=${TMUX_SESSION:-limo}

# Overlay sanity check (non-fatal)
if [ -f "$SCRIPT_DIR/check_overlay.sh" ]; then
  "$SCRIPT_DIR/check_overlay.sh" || true
fi

# Start tmux windows
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install with: sudo apt install tmux" >&2
  exit 1
fi

# Restart session
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

# roscore
tmux new-session -d -s "$SESSION" -n roscore "roscore"
sleep 1

# base
tmux new-window -t "$SESSION" -n base "roslaunch limo_base limo_base.launch"

# lidar (USB Tmini Plus)
tmux new-window -t "$SESSION" -n lidar "roslaunch limo_control sensors/lidar_tmini_usb.launch"

# optional depth camera + pipeline
if [ "$WITH_DEPTH" = "1" ]; then
  tmux new-window -t "$SESSION" -n depth "roslaunch limo_control sensors/orbbec_gemini330_v4l2.launch enable_color:=true enable_depth:=true"
  tmux new-window -t "$SESSION" -n depthpipe "roslaunch limo_control depth_cloud_pipeline.launch scan_frame:=base_link image_topic:=depth/image_raw"
fi

# avoidance (LiDAR-only) or depth avoidance
if [ "$WITH_AVOID" = "1" ]; then
  tmux new-window -t "$SESSION" -n avoid "roslaunch limo_control avoid_only.launch"
fi

# slam (optional)
if [ "$WITH_SLAM" = "1" ]; then
  tmux new-window -t "$SESSION" -n slam "roslaunch limo_control slam.launch"
fi

# rviz (local display). Requires X/desktop; headless setups should run RViz on a remote PC.
if [ "$WITH_RVIZ" = "1" ]; then
  tmux new-window -t "$SESSION" -n rviz "rviz"
fi

# focus on base window
tmux select-window -t "$SESSION":1

echo "Attached to tmux session: $SESSION"
tmux attach -t "$SESSION"
