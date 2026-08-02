"""Per-robot names and conventions for the automation stack.

Select with the robot= kwarg on printerAutomation (threads down through
ArucoDetectionViewer and PoseReader), e.g.:
    node = start_node(sim=1, robot='lite6')

'ar4' is the Annin AR4 (annin_ar4 packages); 'lite6' is the UFACTORY Lite 6
via xarm_ros2 (launch Gazebo with scripts/launchVirtualXArmLite6.sh, which
adds the simulated D435i wrist camera).

NO JOINT VALUES FOR THE HOME POSE LIVE HERE. 'home_state' names a group_state in
the robot's SRDF, and go_home reads that state's joint values off the live
/robot_description_semantic at runtime (see srdf_states.py). The SRDF is
therefore the single source of truth: the same pose RViz offers in its goal-state
dropdown, and the pose the sim spawns at (the matching initial_value params in
each description's ros2_control xacro).
"""

import numpy as np

ROBOT_CONFIGS = {
    'ar4': {
        'joint_names': ["joint_1", "joint_2", "joint_3",
                        "joint_4", "joint_5", "joint_6"],
        'base_link': "base_link",
        'end_effector_link': "link_6",
        'move_group': "ar_manipulator",
        # group_state in annin_ar4_moveit_config/srdf/ar_macro.srdf.xacro
        'home_state': 'home',
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
            # jaw OPENING (m, between the fingertips) at the configured
            # open/closed joint positions, for width-based commands
            # (set_gripper(width=...)). Both jaws travel 0.0145 m
            # (gripper_jaw1/2_joint), so the opening changes by ~0.029 m over
            # the stroke — VERIFY the absolute open value with calipers.
            'open_width': 0.029,
            'closed_width': 0.0,
        },
        # The AR4 URDF has no TCP frame past the jaws, so printerAutomation
        # broadcasts a SYNTHETIC one ('grasp_tcp') from link_6 using
        # tcp_offset, and the same grasp-frame math as the xarm robots then
        # applies. rpy is DERIVED, not tuned: Rz(-90 deg) is exactly what makes
        # the canonical GRASP_ORI_IN_MARKER reproduce the AR4's old hand-tuned
        # offset_ori [0, pi, pi/2], so behavior is unchanged where the old
        # numbers worked. pos is the jaw-tip guess (jaw joints sit ~0.036 m
        # along link_6 Z, fingers extend past that) — VERIFY on hardware and
        # adjust; it only shifts where 'grasp'-frame waypoints land, not the
        # orientation.
        'grasp_frame': 'grasp_tcp',
        'tcp_offset': {'pos': [0.0, 0.0, 0.06],
                       'rpy': [0.0, 0.0, -np.pi / 2]},
        # links allowed to touch an attached (held) object without MoveIt
        # flagging a collision — the jaws are clamped onto it by definition
        'gripper_touch_links': ['gripper_base_link', 'gripper_jaw1_link',
                                'gripper_jaw2_link', 'link_6'],
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
        # group_state in xarm_moveit_config/srdf/_lite6_macro.srdf.xacro
        'home_state': 'home',
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
            # the hardware gripper is binary (open/close services only) —
            # width/fraction commands are thresholded at half travel. Opening
            # per the Lite 6 gripper spec (~16 mm) — VERIFY with calipers.
            'open_width': 0.016,
            'closed_width': 0.0,
        },
        # stock gripper's fingertip frame (uflite_gripper.urdf.xacro puts
        # link_tcp 0.0836 m past the gripper body) — grasped objects anchor here
        'grasp_frame': 'link_tcp',
        'gripper_touch_links': ['uflite_gripper_link', 'uflite_finger1',
                                'uflite_finger2', 'link_tcp', 'link_eef'],
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
        # group_state in xarm_moveit_config/srdf/_xarm6_macro.srdf.xacro
        'home_state': 'home',
        'camera_frame': "camera_color_optical_frame",
        'color_topic': "/camera/color/image_raw",
        'depth_topic': "/camera/depth/image",
        'camera_info_topic': "/camera/color/camera_info",
        'camera_z_offset': 0.04,
        # Stock UFACTORY xArm gripper (add_gripper:=true on the launch). It is
        # exposed as a plain JointTrajectoryController — xarm_gripper_traj_controller
        # over drive_joint — with no GripperCommand action and no driver service,
        # so the 'joint_trajectory' kind publishes trajectory points directly.
        # That is the same mechanism launchVirtualXArm6.sh uses to park the
        # gripper, so it works in sim as well as on hardware.
        # drive_joint: 0 = fully open, 0.85 = fully closed (URDF limit). Open is
        # 0.05 rather than 0: gz settles the joint just under its lower stop and
        # MoveIt's CheckStartStateBounds then rejects every plan (see the
        # GRIPPER PARK note in scripts/launchVirtualXArm6.sh).
        'gripper': {
            'kind': 'joint_trajectory',
            'topic': '/xarm_gripper_traj_controller/joint_trajectory',
            'gripper_joint_names': ["drive_joint"],
            'open_gripper_joint_positions': [0.05],
            'closed_gripper_joint_positions': [0.80],
            # No fixed 'move_time': the trajectory duration is computed as
            # travel / (max_velocity * the robot's speed scale), so the
            # gripper keeps pace with the arm instead of a fixed slow sweep
            # (put 'move_time' back to pin a duration explicitly).
            # max_velocity is drive_joint's URDF limit
            # (xarm_gripper.urdf.xacro velocity="2").
            'max_velocity': 2.0,
            # how long to wait for the joint to actually get there. Generous
            # on purpose: in Gazebo the joint can lag the commanded time
            # badly, and the settle loop already exits early on arrival or
            # stall — this only bounds the worst case.
            'settle_timeout': 25.0,
            # jaw OPENING (m) at the configured open/closed joint positions,
            # for width-based commands (set_gripper(width=...)). Derived from
            # the xArm Gripper's 86 mm stroke over drive_joint 0..0.85 rad,
            # assumed LINEAR: 0.05 rad -> ~81 mm, 0.80 rad -> ~5 mm. The real
            # linkage is slightly nonlinear — VERIFY with calipers at a couple
            # of widths and adjust.
            'open_width': 0.081,
            'closed_width': 0.005,
        },
        # stock gripper's fingertip frame (xarm_gripper.urdf.xacro puts
        # link_tcp 0.172 m past the flange) — grasped objects anchor here, NOT
        # at link_eef, or they overlap the gripper body
        'grasp_frame': 'link_tcp',
        'gripper_touch_links': ['xarm_gripper_base_link',
                                'left_outer_knuckle', 'left_finger',
                                'left_inner_knuckle', 'right_outer_knuckle',
                                'right_finger', 'right_inner_knuckle',
                                'link_tcp', 'link_eef'],
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
