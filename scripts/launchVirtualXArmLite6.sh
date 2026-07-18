#!/usr/bin/env bash
# Gazebo + MoveIt for the UFACTORY Lite 6.
# Needs xarm_ros2 (jazzy branch) built in the workspace:
#   colcon build --packages-up-to xarm_moveit_config xarm_gazebo

source ~/ar4_ws/install/setup.bash

# a stale gz server from a previous run holds the 'default' world topics and
# makes the new launch hang waiting for /world/default/create; clear it first
for p in $(ps -eo pid,comm | grep -iE "gz-sim|ruby" | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null
done
sleep 1

# add_realsense_d435i mounts a simulated D435i on the wrist and bridges
# /camera/color/image_raw + camera_info + /camera/depth/image, which the
# automation stack's 'lite6' robot config subscribes to.
# load_table:=false spawns the robot at the origin in an empty world
# (patched into xarm_gazebo/_robot_beside_table_gazebo.launch.py).
ros2 launch xarm_moveit_config lite6_moveit_gazebo.launch.py \
    add_realsense_d435i:=true load_table:=false
