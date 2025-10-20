#!/usr/bin/env bash
set -euo pipefail

# Simple bringup helper for base + LiDAR on Noetic

if [ -f /opt/ros/noetic/setup.bash ]; then
  source /opt/ros/noetic/setup.bash
fi
if [ -f "$HOME/limo_ws/devel/setup.bash" ]; then
  source "$HOME/limo_ws/devel/setup.bash"
fi

USE_TMUX=${USE_TMUX:-1}

if [ "${USE_TMUX}" = "1" ] && command -v tmux >/dev/null 2>&1; then
  session=${TMUX_SESSION:-limo}
  tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session"
  tmux new-session -d -s "$session" -n roscore "roscore"
  sleep 1
  tmux new-window -t "$session" -n base "roslaunch limo_base limo_base.launch"
  tmux new-window -t "$session" -n lidar "roslaunch limo_control sensors/lidar_tmini_usb.launch"
  tmux select-window -t "$session":1
  echo "Attached to tmux session: $session"
  tmux attach -t "$session"
else
  echo "Run these in separate terminals:"
  cat <<'CMDS'
roscore
roslaunch limo_base limo_base.launch
roslaunch limo_control sensors/lidar_tmini_usb.launch
CMDS
fi

