#!/usr/bin/env bash
# Gazebo + MoveIt for the UFACTORY Lite 6.
# Needs xarm_ros2 (jazzy branch) built in the workspace:
#   colcon build --packages-up-to xarm_moveit_config xarm_gazebo

source ~/ar4_ws/install/setup.bash

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
# add_gripper:=true appends the UFACTORY Lite gripper at link_eef
# (uflite_gripper_macro in xarm_description/urdf/xarm_device_macro.xacro);
# without it the arm is built bare to the flange and no gripper exists.
ros2 launch xarm_moveit_config lite6_moveit_gazebo.launch.py \
    add_realsense_d435i:=true add_gripper:=true load_table:=false
