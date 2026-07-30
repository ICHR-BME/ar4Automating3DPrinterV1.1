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
ROBOT_IP="192.168.0.181"
#ROBOT_IP="192.168.0.150"

# add_gripper:=true appends the UFACTORY Lite gripper at link_eef and brings up
# its controller; must match how the arm is physically tooled.
ros2 launch xarm_moveit_config lite6_moveit_realmove.launch.py \
    robot_ip:="${ROBOT_IP}" \
    add_realsense_d435i:=true \
    add_gripper:=true
