#!/usr/bin/env bash
# Gazebo + MoveIt for the UFACTORY Lite 6.
# Needs xarm_ros2 (jazzy branch) built in the workspace:
#   colcon build --packages-up-to xarm_moveit_config xarm_gazebo

source ~/ar4_ws/install/setup.bash

ros2 launch xarm_moveit_config lite6_moveit_gazebo.launch.py
