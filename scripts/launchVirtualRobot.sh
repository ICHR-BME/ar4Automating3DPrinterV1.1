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


# ---- DEFAULT SPEED -------------------------------------------------------
# MoveIt's built-in default_velocity/acceleration_scaling_factor is 0.1 (10%)
# for any plan request that doesn't set its own scaling. Raise move_group's
# defaults to 50% once it is up (the RViz panel's spinboxes are set to 0.5 in
# the .rviz configs separately). Background, exits on its own.
(
    deadline=$((SECONDS + 120))
    until ros2 param set /move_group default_velocity_scaling_factor 0.5 2>/dev/null \
            | grep -q successful; do
        [ "$SECONDS" -ge "$deadline" ] && \
            echo "DEFAULT SPEED: move_group never came up, skipping" && exit 1
        sleep 2
    done
    ros2 param set /move_group default_acceleration_scaling_factor 0.5 >/dev/null 2>&1
    echo "DEFAULT SPEED: move_group scaling defaults set to 0.5"
) &

ros2 launch annin_ar4_moveit_config moveit.launch.py use_sim_time:=true include_gripper:=True
