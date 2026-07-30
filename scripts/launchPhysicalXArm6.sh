#!/usr/bin/env bash
# MoveIt + real hardware for the UFACTORY xArm 6 (physical robot).
# Physical-hardware counterpart of launchVirtualXArm6.sh — same robot
# config ('xarm6') and same camera topics, but the arm is the real controller
# and the D435i frames come from the real RealSense driver instead of gz.
# Needs xarm_ros2 (jazzy branch) built in the workspace:
#   colcon build --packages-up-to xarm_moveit_config
# and the realsense2_camera package (already installed under /opt/ros).

source ~/ar4_ws/install/setup.bash

# ---- config (edit these; no CLI args) ------------------------------------
# The xArm 6 control box's IP on the current network. The arm must be powered,
# enabled, and reachable (ping it first). This is the one thing the sim script
# never needed.
ROBOT_IP="192.168.1.226"

# add_gripper:=true appends the xArm gripper at link_eef and brings up its
# controller; must match how the arm is physically tooled.
ros2 launch xarm_moveit_config xarm6_moveit_realmove.launch.py \
    robot_ip:="${ROBOT_IP}" \
    add_realsense_d435i:=true \
    add_gripper:=true
