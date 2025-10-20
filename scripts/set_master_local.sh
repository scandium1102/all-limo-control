#!/usr/bin/env bash
set -euo pipefail

# Derive Jetson's primary IP automatically
JETSON_IP=${ROS_IP:-$(ip route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++){if($i=="src"){print $(i+1); exit}}}')}
: "${JETSON_IP:?Failed to derive local IP. Set ROS_IP manually.}"

export ROS_MASTER_URI=${ROS_MASTER_URI:-http://$JETSON_IP:11311}
export ROS_IP=$JETSON_IP
export ROS_HOSTNAME=$JETSON_IP

echo "ROS_MASTER_URI=$ROS_MASTER_URI"
echo "ROS_IP=$ROS_IP"
echo "ROS_HOSTNAME=$ROS_HOSTNAME"
