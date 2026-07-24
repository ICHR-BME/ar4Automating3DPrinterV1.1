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
        # 'moveit_action' kind: driven through pymoveit2's GripperInterface
        # (a GripperCommand action). The remaining keys are its constructor args.
        'gripper': {
            'kind': 'moveit_action',
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
        # with the flipped offset_ori below the D435i hangs ~4 cm below
        # link_eef, so raise the EEF to keep the marker centered (mirror of the
        # -0.04 that the camera-above roll needed; re-check on the first scans)
        #'camera_z_offset': 0.04,
        #It is 0.06 for the x arm lite 6 custom gripper
        'camera_z_offset': 0.06,
        # Stock UFACTORY Lite 6 gripper. On real hardware it is NOT a MoveIt
        # GripperCommand controller (that path is sim/fake only) — the xarm
        # driver exposes empty-request Call services instead. 'lite6_service'
        # kind routes open_gripper/close_gripper to these. In sim the gripper is
        # force-disabled (make_sim_node), so these services are only used on hw.
        # NOTE: namespace is 'ufactory' (hw_ns default in lite6_moveit_realmove
        # launch), NOT 'xarm'. The services must also be enabled in
        # xarm_user_params.yaml (open/close_lite6_gripper: true) — off by default.
        'gripper': {
            'kind': 'lite6_service',
            'open_service': '/ufactory/open_lite6_gripper',
            'close_service': '/ufactory/close_lite6_gripper',
        },
        # good frame == base frame for the lite6 (robot spawns at the world
        # origin with zero yaw, so no AR4-style 90 deg convention)
        'frame_rotation_angles': np.array([0.0, 0.0, 0.0]),
        'frame_offset_angles': np.array([0.0, 0.0, 0.0]),
        # AR4 value rolled 180 deg about the tool approach axis: the D435i is
        # mounted on the opposite side of the eef here, so the unrolled AR4
        # orientation put the camera above the gripper instead of below it.
        # Same tool Z (still faces the marker), only the roll differs.
        'offset_ori': np.array([np.pi, 0.0, np.pi]),
    },
    # UFACTORY xArm 6 via xarm_ros2 (launch Gazebo with
    # scripts/launchVirtualXArm6.sh). Shares the Lite 6's kinematic naming
    # (joint1..joint6, link_base, link_eef) and the same simulated D435i wrist
    # camera (add_realsense_d435i:=true bridges the identical /camera/* topics
    # and camera_color_optical_frame), so this is the lite6 block with the
    # move_group swapped. The camera_z_offset/offset_ori are copied from the
    # lite6 and MAY need re-tuning on the first scans: the xArm 6 has a longer
    # reach (~0.7 m) and a different wrist, so the D435i's offset from link_eef
    # is not guaranteed to match the Lite 6's.
    'xarm6': {
        'joint_names': ["joint1", "joint2", "joint3",
                        "joint4", "joint5", "joint6"],
        'base_link': "link_base",
        'end_effector_link': "link_eef",
        'move_group': "xarm6",
        'camera_frame': "camera_color_optical_frame",
        'color_topic': "/camera/color/image_raw",
        'depth_topic': "/camera/depth/image",
        'camera_info_topic': "/camera/color/camera_info",
        'camera_z_offset': 0.04,
        # no gripper wired up yet: gripper commands are skipped
        'gripper': None,
        # good frame == base frame (robot spawns at the world origin, zero yaw)
        'frame_rotation_angles': np.array([0.0, 0.0, 0.0]),
        'frame_offset_angles': np.array([0.0, 0.0, 0.0]),
        'offset_ori': np.array([np.pi, 0.0, np.pi]),
    },
}


def get_robot_config(robot):
    try:
        return ROBOT_CONFIGS[robot]
    except KeyError:
        raise ValueError(
            f"Unknown robot '{robot}'. Available: {list(ROBOT_CONFIGS)}")
