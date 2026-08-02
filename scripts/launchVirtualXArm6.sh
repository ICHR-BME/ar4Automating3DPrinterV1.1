#!/usr/bin/env bash
# Gazebo + MoveIt for the UFACTORY xArm 6.
# Needs xarm_ros2 (jazzy branch) built in the workspace:
#   colcon build --packages-up-to xarm_moveit_config xarm_gazebo

source ~/ar4_ws/install/setup.bash

# ---- config (edit these; no CLI args) ------------------------------------
# Where to park the gripper once the sim is up, in rad on drive_joint
# (0 = fully open, 0.85 = fully closed). Must be > 0; see GRIPPER PARK below.
GRIPPER_PARK_POSITION=0.05

# The headless gz Sensors system renders the camera via EGL device platform and
# grabs EGL device[0] = the NVIDIA RTX 3050 (pci 01:00.0, renderD129) since it
# sorts before the AMD 780M (63:00.0). Under Mesa's libEGL that device can't be
# driven ("failed to create dri2 screen"), which cascades into the gz-sensors
# render thread aborting (Ogre material/scene identity crash) — killing the gz
# server so controllers never spawn and joint_states never publishes. Pin glvnd
# to NVIDIA's OWN EGL vendor so the offscreen render uses the 3050 through the
# proper NVIDIA headless path (no Mesa/DRI2). The GUI and rviz render via GLX on
# the AMD X display and are unaffected by this EGL vendor pin.
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Pin gz to loopback ONLY on hosts where the default interface can't talk to
# itself (e.g. wifi handing out CGNAT 100.64/10 addresses that Tailscale's
# firewall drops); healthy machines keep gz's stock behavior. Detection logic
# and the full story live in detect_gz_ip.py.
GZ_IP_AUTO=$(python3 "$(dirname "$0")/detect_gz_ip.py")
[ -n "$GZ_IP_AUTO" ] && export GZ_IP="$GZ_IP_AUTO"

# a stale gz server from a previous run holds the 'default' world topics and
# makes the new launch hang waiting for /world/default/create; clear it first
for p in $(ps -eo pid,comm | grep -iE "gz-sim|ruby" | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null
done
sleep 1

# add_realsense_d435i mounts a simulated D435i on the wrist and bridges
# /camera/color/image_raw + camera_info + /camera/depth/image, which the
# automation stack's 'xarm6' robot config subscribes to.
# load_table:=false spawns the robot at the origin in an empty world
# (patched into xarm_gazebo/_robot_beside_table_gazebo.launch.py).
# add_gripper:=true appends the xArm gripper at link_eef (xarm_gripper_macro in
# xarm_description/urdf/xarm_device_macro.xacro); without it the arm is built
# bare to the flange and no gripper geometry or controller exists.
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

# ---- GRIPPER PARK --------------------------------------------------------
# gz spawns drive_joint on its lower stop (0 = fully open) and it settles a
# fraction below, so MoveIt's CheckStartStateBounds adapter rejects the start
# state and every gripper plan from rviz aborts with 'Start state out of
# bounds' -- including closing it. Jazzy dropped the old
# start_state_max_bounds_error tolerance, so there is no margin to absorb it.
# Widening the URDF limit does not help: the joint just settles on whatever the
# new lower stop is. One explicit command is what fixes it -- the joint tracks
# it exactly and stays in bounds from then on. Runs in the background and
# exits on its own; the launch below stays in the foreground as usual.
(
    deadline=$((SECONDS + 120))
    until ros2 control list_controllers 2>/dev/null \
            | grep -q "xarm_gripper_traj_controller.*active"; do
        [ "$SECONDS" -ge "$deadline" ] && \
            echo "GRIPPER PARK: gripper controller never came up, skipping" && exit 1
        sleep 2
    done
    sleep 2
    ros2 topic pub --once /xarm_gripper_traj_controller/joint_trajectory \
        trajectory_msgs/msg/JointTrajectory \
        "{joint_names: ['drive_joint'], points: [{positions: [${GRIPPER_PARK_POSITION}], time_from_start: {sec: 2}}]}" \
        >/dev/null 2>&1
    echo "GRIPPER PARK: gripper parked at ${GRIPPER_PARK_POSITION} rad"
) &

ros2 launch xarm_moveit_config xarm6_moveit_gazebo.launch.py \
    add_realsense_d435i:=true add_gripper:=true load_table:=false
