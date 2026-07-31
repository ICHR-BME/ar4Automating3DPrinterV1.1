#!/usr/bin/env bash

# gz-transport binds to the first non-loopback interface; on this wifi that is a
# CGNAT address inside 100.64.0.0/10, which Tailscale's ts-input chain DROPs when
# it does not arrive on tailscale0 -- so gz service calls between the local gz
# processes hang forever. Pin gz to loopback. See launchVirtualXArm6.sh.
export GZ_IP=127.0.0.1

ros2 launch annin_ar4_gazebo gazebo.launch.py
sleep 6

# Camera is now included directly in the world file (empty.world)
# No need to spawn it separately

sleep 2

ros2 launch annin_ar4_moveit_config moveit.launch.py use_sim_time:=true include_gripper:=True


