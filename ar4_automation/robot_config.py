"""Per-robot names and conventions for the automation stack.

Select with the robot= kwarg on printerAutomation (threads down through
ArucoDetectionViewer and PoseReader), e.g.:
    node = start_node(sim=1, robot='lite6')

'ar4' is the Annin AR4 (annin_ar4 packages); 'lite6' is the UFACTORY Lite 6
via xarm_ros2 (launch Gazebo with scripts/launchVirtualXArmLite6.sh, which
adds the simulated D435i wrist camera).
"""

import numpy as np

ROBOT_CONFIGS = {
    'ar4': {
        'joint_names': ["joint_1", "joint_2", "joint_3",
                        "joint_4", "joint_5", "joint_6"],
        'base_link': "base_link",
        'end_effector_link': "link_6",
        'move_group': "ar_manipulator",
        'camera_frame': "ee_camera_link",
        'color_topic': "/rgbd_camera/image",
        'depth_topic': "/rgbd_camera/depth_image",
        'camera_info_topic': "/rgbd_camera/camera_info",
        # camera sits below the gripper; raise the EE this much (m) when scanning
        'camera_z_offset': 0.06,
        'gripper': {
            'gripper_joint_names': ["gripper_jaw1_joint"],
            'open_gripper_joint_positions': [0.00],
            'closed_gripper_joint_positions': [0.0145],
            'gripper_group_name': "ar_gripper",
            'gripper_command_action_name': "gripper_controller/gripper_cmd",
        },
        # bad frame (base_link) -> good frame rotation, and the euler offset of
        # the neutral tool orientation (calibrated for the AR4)
        'frame_rotation_angles': np.array([0.0, 0.0, np.pi / 2]),
        'frame_offset_angles': np.array([-0.6162, -1.5706, -2.1870]),
        # tool orientation used to face a marker (bad-frame euler XYZ)
        'offset_ori': np.array([0.0, np.pi, np.pi / 2]),
    },
    'lite6': {
        'joint_names': ["joint1", "joint2", "joint3",
                        "joint4", "joint5", "joint6"],
        'base_link': "link_base",
        'end_effector_link': "link_eef",
        'move_group': "lite6",
        # simulated RealSense D435i (add_realsense_d435i:=true on the launch)
        'camera_frame': "camera_color_optical_frame",
        'color_topic': "/camera/color/image_raw",
        'depth_topic': "/camera/depth/image",
        'camera_info_topic': "/camera/color/camera_info",
        # D435i sits ~4 cm above link_eef on this mount: aim the EEF lower so
        # the marker lands centered instead of cut off at the bottom of frame
        # (tuned in sim; -0.07 and below makes low viewing poses unreachable)
        'camera_z_offset': -0.04,
        # no gripper wired up yet: gripper commands are skipped
        'gripper': None,
        # good frame == base frame for the lite6 (robot spawns at the world
        # origin with zero yaw, so no AR4-style 90 deg convention)
        'frame_rotation_angles': np.array([0.0, 0.0, 0.0]),
        'frame_offset_angles': np.array([0.0, 0.0, 0.0]),
        # UNTUNED: tool-vs-marker orientation copied from the AR4 (depends on
        # the camera/tool mounting, not the base frame); verify on first scans
        'offset_ori': np.array([0.0, np.pi, np.pi / 2]),
    },
}


def get_robot_config(robot):
    try:
        return ROBOT_CONFIGS[robot]
    except KeyError:
        raise ValueError(
            f"Unknown robot '{robot}'. Available: {list(ROBOT_CONFIGS)}")
