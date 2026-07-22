#!/usr/bin/env bash
# MoveIt + real hardware for the UFACTORY Lite 6 (physical robot).
# Physical-hardware counterpart of launchVirtualXArmLite6.sh — same robot
# config ('lite6') and same camera topics, but the arm is the real controller
# and the D435i frames come from the real RealSense driver instead of gz.
# Needs xarm_ros2 (jazzy branch) built in the workspace:
#   colcon build --packages-up-to xarm_moveit_config
# and the realsense2_camera package (already installed under /opt/ros).

source ~/ar4_ws/install/setup.bash

# ---- config (edit these; no CLI args) ------------------------------------
# The Lite 6 control box's IP on the current network. The arm must be powered,
# enabled, and reachable (ping it first). This is the one thing the sim script
# never needed.
ROBOT_IP="192.168.0.150"
# --------------------------------------------------------------------------

# The 'lite6' automation config subscribes to /camera/color/image_raw,
# /camera/color/camera_info and /camera/depth/image (robot_config.py). In sim
# the gz bridge published those; here the real realsense2_camera driver does,
# and add_realsense_d435i:=true keeps the D435i links in the URDF so
# camera_color_optical_frame is published by robot_state_publisher (TF), which
# is why the camera driver runs with publish_tf:=false below — one owner per
# frame, no duplicate/ fighting TF.

# clean up the background camera driver when this script exits (Ctrl-C on the
# foreground MoveIt launch tears the whole thing down)
CAM_PID=""
cleanup() {
    [ -n "$CAM_PID" ] && kill "$CAM_PID" 2>/dev/null
}
trap cleanup EXIT INT TERM

# Real RealSense D435i driver. Run the node directly (not rs_launch.py) so the
# topics land under /camera with the exact names the 'lite6' config expects:
#   /camera/color/image_raw, /camera/color/camera_info  (aruco needs these)
# and remap the driver's depth (depth/image_rect_raw) to the /camera/depth/image
# name the sim used. publish_tf:=false — the URDF owns the camera frames.
ros2 run realsense2_camera realsense2_camera_node --ros-args \
    -r __ns:=/camera \
    -p enable_color:=true \
    -p enable_depth:=true \
    -p align_depth.enable:=true \
    -p publish_tf:=false \
    -r /camera/depth/image_rect_raw:=/camera/depth/image &
CAM_PID=$!

# give the camera a moment to come up before MoveIt starts planning-scene work
sleep 3

# Real robot driver + MoveIt for the Lite 6. add_realsense_d435i:=true mirrors
# the sim launch so the URDF/SRDF (frames, self-collision) match exactly.
ros2 launch xarm_moveit_config lite6_moveit_realmove.launch.py \
    robot_ip:="${ROBOT_IP}" \
    add_realsense_d435i:=true
