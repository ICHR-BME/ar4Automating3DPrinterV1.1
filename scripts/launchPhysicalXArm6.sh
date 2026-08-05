#!/usr/bin/env bash
set -euo pipefail

# Physical xArm 6 + MoveIt + RealSense.  The controller has been positively
# identified as xArm 6 by gen_kinematics_params.py; do not substitute Lite 6.
XARM_WS="${XARM_WS:-$HOME/dev_ws}"
ROBOT_IP="${ROBOT_IP:?Set ROBOT_IP to the xArm control-box IP}"
KINEMATICS_SUFFIX="${KINEMATICS_SUFFIX:-fernanda_xarm6}"

source /opt/ros/humble/setup.bash
source "$XARM_WS/install/local_setup.bash"

CAM_PID=""
cleanup() {
    [ -z "$CAM_PID" ] || kill "$CAM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ros2 run realsense2_camera realsense2_camera_node --ros-args \
    -r __ns:=/camera \
    -p enable_color:=true \
    -p enable_depth:=true \
    -p align_depth.enable:=true \
    -p publish_tf:=false \
    -r /camera/depth/image_rect_raw:=/camera/depth/image &
CAM_PID=$!
sleep 3

ros2 launch xarm_moveit_config _robot_moveit_realmove.launch.py \
    robot_ip:="$ROBOT_IP" \
    robot_type:=xarm \
    dof:=6 \
    hw_ns:=xarm \
    kinematics_suffix:="$KINEMATICS_SUFFIX" \
    add_realsense_d435i:=true
