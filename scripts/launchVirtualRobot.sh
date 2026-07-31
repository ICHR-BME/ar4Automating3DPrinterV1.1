#!/usr/bin/env bash

# Pin gz to loopback ONLY on hosts where the default interface can't talk to
# itself (e.g. wifi handing out CGNAT 100.64/10 addresses that Tailscale's
# firewall drops); healthy machines keep gz's stock behavior. Detection logic
# and the full story live in detect_gz_ip.py.
GZ_IP_AUTO=$(python3 "$(dirname "$0")/detect_gz_ip.py")
[ -n "$GZ_IP_AUTO" ] && export GZ_IP="$GZ_IP_AUTO"

ros2 launch annin_ar4_gazebo gazebo.launch.py
sleep 6

# Camera is now included directly in the world file (empty.world)
# No need to spawn it separately

sleep 2

ros2 launch annin_ar4_moveit_config moveit.launch.py use_sim_time:=true include_gripper:=True


