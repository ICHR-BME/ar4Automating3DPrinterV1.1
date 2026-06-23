#!/usr/bin/env python3

import sys
import time
import warnings
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Point, Quaternion, Pose
from std_msgs.msg import String
from tf_transformations import quaternion_from_euler, euler_from_quaternion

import math

import tf2_ros
from rclpy.time import Time

def quat_to_euler(x: float, y: float, z: float, w: float):
	roll, pitch, yaw = R.from_quat([x, y, z, w]).as_euler("XYZ", degrees=False)
	return roll, pitch, yaw

# MoveItErrorCodes values -> human-readable reason (for diagnosing STATUS_ABORTED).
_MOVEIT_ERR = {
	1: "SUCCESS", 99999: "UNDEFINED", -1: "FAILURE", -2: "PLANNING_FAILED",
	-3: "INVALID_MOTION_PLAN", -4: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
	-5: "CONTROL_FAILED", -6: "UNABLE_TO_AQUIRE_SENSOR_DATA", -7: "TIMED_OUT",
	-8: "PREEMPTED", -10: "START_STATE_IN_COLLISION", -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
	-12: "GOAL_IN_COLLISION", -13: "GOAL_VIOLATES_PATH_CONSTRAINTS", -14: "GOAL_CONSTRAINTS_VIOLATED",
	-15: "INVALID_GROUP_NAME", -16: "INVALID_GOAL_CONSTRAINTS", -17: "INVALID_ROBOT_STATE",
	-18: "INVALID_LINK_NAME", -19: "INVALID_OBJECT_NAME", -21: "NO_IK_SOLUTION",
}

def _moveit_err_str(moveit2):
	"""Return a readable MoveIt error-code string for the last execution, or '?'."""
	try:
		code = moveit2.get_last_execution_error_code()
		val = getattr(code, "val", code)
		return f"{_MOVEIT_ERR.get(val, 'UNKNOWN')}({val})"
	except Exception:
		return "?"

import sys
sys.path.insert(0, "/home/koghalai/ar4_ws/src/ar4Automating3DPrinter")
from moveit2 import MoveIt2

class PoseReader(Node):
	"""ROS 2 node that prints the gripper pose every second using pymoveit2."""

	def __init__(self, node_name: Optional[str] = None, enable_pose_print: bool = True):
		super().__init__(node_name or "gripper_pose_reader")

		# Robot specifics (AR4 / MoveIt config defaults)
		joint_names = [
			"joint_1",
			"joint_2",
			"joint_3",
			"joint_4",
			"joint_5",
			"joint_6",
		]
		base_link_name = "base_link"
		end_effector_name = "link_6"  # AR4 end-effector link
		group_name = "ar_manipulator"  # AR4 MoveIt group

		# Initialize MoveIt2 helper
		self._cb_group = ReentrantCallbackGroup()
		self.moveit2 = MoveIt2(
			node=self,
			joint_names=joint_names,
			base_link_name=base_link_name,
			end_effector_name=end_effector_name,
			group_name=group_name,
			use_move_group_action=True,  # Use action instead of service
			callback_group=self._cb_group,
		)

		# Default speed limits (0.0 causes MoveIt to warn and fall back to 1.0)
		self.moveit2.max_velocity = 0.9
		self.moveit2.max_acceleration = 0.9

		# Seconds to pause after motion completes so MoveIt's TrajectoryExecutionManager
		# fully releases before the next command is sent.
		self.move_settle_delay = 0.5


		# Keep local references for frames
		self.base_link_name = base_link_name
		self.end_effector_name = end_effector_name

		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

		# Subscribe directly to joint states
		self._last_joint_msg = None  # list[float] ordered by self.moveit2.joint_names
		self._fk_future = None
		# Hardcoded baseline orientation (home) so printed rpy is (0,0,0) at home
		self.pose = np.array([-1,-1,-1,-1,-1,-1])
		self.quat = np.array([-1, -1, -1, -1])
		self.frame = ""

		self.enable_pose_print = enable_pose_print

		self.create_subscription(
			JointState,
			"joint_states",
			self._on_joint_states,
			10,
		)
		# Always update pose every 0.5s, but only print if enabled
		self._timer = self.create_timer(0.5, self._on_timer)

		self.get_logger().info(
			f"PoseReader started; base='{base_link_name}', eef='{end_effector_name}'"
		)
		self.frameRotationAngles = np.array([0, 0, np.pi/2])  # Rotation from Bad Frame to Good Frame
		self.frameOffsetAngles = np.array([-0.6162, -1.5706, -2.1870])  # No translation offset

		# Publisher used to cancel any in-progress MoveIt trajectory before sending a new one.
		self._cancellation_pub = self.create_publisher(String, '/trajectory_execution_event', 1)

	def _cancel_and_wait(self, wait_timeout=3.0):
		"""Cancel any in-flight MoveIt trajectory and wait for execution to stop.

		Publishes a stop event, then waits up to *wait_timeout* seconds for
		MoveIt's result callback to fire (which clears __is_executing).  After
		the wait the flags are force-cleared so the next goal starts from a
		clean state, followed by a brief pause for MoveIt's
		TrajectoryExecutionManager to fully release the trajectory lock.
		"""
		_stop = String()
		_stop.data = 'stop'
		self._cancellation_pub.publish(_stop)
		# Wait for any active trajectory's result callback to fire naturally
		# (it will set __is_executing = False).  This prevents the next goal
		# from arriving while MoveIt is still busy.
		_deadline = time.time() + wait_timeout
		while (getattr(self.moveit2, '_MoveIt2__is_executing', False) or
		       getattr(self.moveit2, '_MoveIt2__is_motion_requested', False)):
			if time.time() > _deadline:
				break
			time.sleep(0.05)
		# Force-clear flags regardless (handles the case where the action server
		# never sent a result, e.g. after a hard timeout or server restart).
		self.moveit2._MoveIt2__is_motion_requested = False
		self.moveit2._MoveIt2__is_executing = False
		# Brief pause so TrajectoryExecutionManager has released the lock.
		time.sleep(0.3)

	def move_to_configuration(self, joint_positions, timeout=15.0, max_retries=2):
		"""Move to *joint_positions* with automatic retry on transient failures.

		Mirrors the move_to_pose() pattern: cancels any in-flight trajectory
		before each attempt and polls the MoveIt flags with a deadline instead
		of calling wait_until_executed() (which has no timeout).

		Returns True if the motion succeeded, False if all attempts failed.
		"""
		for attempt in range(max_retries + 1):
			if attempt > 0:
				self.get_logger().warn(
					f"[move_to_configuration] Retry {attempt}/{max_retries}…"
				)

			self._cancel_and_wait()
			self.moveit2.motion_suceeded = False

			self.moveit2.move_to_configuration(joint_positions=joint_positions)

			_deadline = time.time() + timeout
			timed_out = False
			while (getattr(self.moveit2, '_MoveIt2__is_motion_requested', False) or
			       getattr(self.moveit2, '_MoveIt2__is_executing', False)):
				if time.time() > _deadline:
					self.get_logger().error(
						f"[move_to_configuration] timed out after {timeout}s."
					)
					timed_out = True
					self._cancel_and_wait()
					break
				time.sleep(0.05)

			if self.moveit2.motion_suceeded:
				time.sleep(self.move_settle_delay)
				return True

			reason = f"timed out after {timeout}s" if timed_out else "motion aborted/failed"
			err = _moveit_err_str(self.moveit2)
			if attempt < max_retries:
				self.get_logger().warn(
					f"[move_to_configuration] {reason} (MoveIt err={err}) on attempt {attempt + 1} — retrying…"
				)
			else:
				self.get_logger().error(
					f"[move_to_configuration] {reason} (MoveIt err={err}) — all attempts exhausted."
				)

		return False

	def move_to_pose(self, pos, euler, max_retries=2):
		"""Move to *pos* / *euler* with automatic retry on transient failures.

		A "transient failure" is one where MoveIt aborted or timed out because
		it was still busy executing a previous trajectory.  On failure the
		in-flight trajectory is properly cancelled before the next attempt so
		that MoveIt is in a clean idle state.

		Returns True if the motion succeeded, False if all attempts failed.
		"""
		bad_pos, bad_euler = self.to_bad_frame(pos, euler)
		q = R.from_euler("XYZ", bad_euler, degrees=False).as_quat()  # [x, y, z, w]
		q_msg = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
		self.get_logger().warn(
			f"[move_to_pose] MoveIt target in base_link: "
			f"pos=[{bad_pos[0]:.4f}, {bad_pos[1]:.4f}, {bad_pos[2]:.4f}] "
			f"quat=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]"
		)

		_timeout = 15.0

		for attempt in range(max_retries + 1):
			if attempt > 0:
				self.get_logger().warn(
					f"[move_to_pose] Retry {attempt}/{max_retries}…"
				)

			# Cancel any in-flight trajectory and wait for MoveIt to go idle
			# before sending a new goal.  This is the key fix: without waiting
			# for the previous trajectory to finish, the new goal is aborted
			# immediately with STATUS_ABORTED.
			self._cancel_and_wait()
			# Reset the success flag so a stale result from a previous
			# trajectory cannot be mistaken for this attempt's outcome.
			self.moveit2.motion_suceeded = False

			self.moveit2.move_to_pose(
				position=Point(x=bad_pos[0], y=bad_pos[1], z=bad_pos[2]),
				quat_xyzw=q_msg,
			)

			# Poll both flags until the action server reports a final result.
			# (wait_until_executed() can return early under MultiThreadedExecutor
			# because the goal-accepted callback clears __is_motion_requested
			# before wait_until_executed checks it.)
			_deadline = time.time() + _timeout
			timed_out = False
			while (getattr(self.moveit2, '_MoveIt2__is_motion_requested', False) or
			       getattr(self.moveit2, '_MoveIt2__is_executing', False)):
				if time.time() > _deadline:
					self.get_logger().error(
						f"[move_to_pose] timed out after {_timeout}s waiting for trajectory to finish."
					)
					timed_out = True
					break
				time.sleep(0.05)

			if self.moveit2.motion_suceeded:
				time.sleep(self.move_settle_delay)
				return True

			reason = f"timed out after {_timeout}s" if timed_out else "motion aborted/failed"
			err = _moveit_err_str(self.moveit2)
			if attempt < max_retries:
				self.get_logger().warn(
					f"[move_to_pose] {reason} (MoveIt err={err}) on attempt {attempt + 1} — retrying…"
				)
			else:
				self.get_logger().error(
					f"[move_to_pose] {reason} (MoveIt err={err}) — planning may have been unsuccessful."
				)

		return False


	def to_good_frame(self, bad_position, bad_euler_angles):
		# Transformation from Bad Frame to Good Frame (BF to GF)

		R_BF_GF_Vec = R.from_euler("XYZ", self.frameRotationAngles, degrees=False)
		R_BF_GF = R_BF_GF_Vec.as_matrix()
		H_BF_GF = np.eye(4)
		H_BF_GF[:3, :3] = R_BF_GF

		# Create rotation matrix from Euler angles
		RBadFrameVec = R.from_euler("XYZ", bad_euler_angles, degrees=False)
		RBadFrame = RBadFrameVec.as_matrix()
		HBadFrame = np.eye(4)
		HBadFrame[:3, :3] = RBadFrame
		HBadFrame[:3, 3] = bad_position

		HGoodFrame = H_BF_GF @ HBadFrame
		good_position = HGoodFrame[:3, 3]

		# Extract rotation matrix and convert to Euler angles ("XYZ" order)
		good_euler_angles_vec = R.from_matrix(HGoodFrame[:3, :3])


		good_euler_angles = good_euler_angles_vec.as_euler("XYZ", degrees=False)
		good_euler_angles -= self.frameOffsetAngles

		return good_position, good_euler_angles

	def to_bad_frame(self, good_position, good_euler_angles):
		"""
		Inverse transformation: from Good Frame back to Bad Frame.
		Args:
			good_position: np.array([x, y, z]) in Good Frame
			good_euler_angles: np.array([roll, pitch, yaw]) in Good Frame ("XYZ" order)
		Returns:
			bad_position, bad_euler_angles in Bad Frame
		"""
		# Inverse rotation from Good Frame to Bad Frame
		R_BF_GF_Vec = R.from_euler("XYZ", self.frameRotationAngles, degrees=False)
		R_GF_BF = R_BF_GF_Vec.as_matrix().T
		H_GF_BF = np.eye(4)
		H_GF_BF[:3, :3] = R_GF_BF

		# Create rotation matrix from Euler angles in Good Frame
		# Add back the offset angles that were subtracted in to_good_frame
		good_euler_angles_corrected = good_euler_angles + self.frameOffsetAngles
		RGoodFrameVec = R.from_euler("XYZ", good_euler_angles_corrected, degrees=False)
		RGoodFrame = RGoodFrameVec.as_matrix()
		HGoodFrame = np.eye(4)
		HGoodFrame[:3, :3] = RGoodFrame
		HGoodFrame[:3, 3] = good_position

		# Apply inverse transformation
		HBadFrame = H_GF_BF @ HGoodFrame
		bad_position = HBadFrame[:3, 3]

		# Extract rotation matrix and convert to Euler angles ("XYZ" order)
		bad_euler_angles_vec = R.from_matrix(HBadFrame[:3, :3])
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", UserWarning)
			bad_euler_angles = bad_euler_angles_vec.as_euler("XYZ", degrees=False)
		return bad_position, bad_euler_angles


	def _on_joint_states(self, msg: JointState):
		# Store joints mapped to the planning group order
		self.jointAngles = msg.position[2:8]
		self.linkNames = msg.name[2:8]

		try:
			self._last_joint_msg = [
				float(msg.position[msg.name.index(j)]) for j in self.moveit2.joint_names
			]
		except ValueError:
			# Missing joints in this message; skip
			return
	
	def get_frame(self, frame = "ee_link"):
		try:
			temp = self.tf_buffer.lookup_transform(
				"world",  # target (base)
				frame,                  # source (camera/link)
				Time()
			)
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
			self.get_logger().warn(f"TF Error: {e}")
			return self.pose

		position = temp.transform.translation
		x, y, z = position.x, position.y, position.z
		
		# Extract rotation (quaternion)
		rotation = temp.transform.rotation
		qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w
		
		# Convert quaternion to Euler angles (roll, pitch, yaw)
		roll, pitch, yaw = quat_to_euler(qx, qy, qz, qw)

		good_pos, good_euler = self.to_good_frame(np.array([x, y, z]), np.array([roll, pitch, yaw]))

		bad_pos, bad_euler = self.to_bad_frame(good_pos, good_euler)
		goodPose = np.array([good_pos[0], good_pos[1], good_pos[2], good_euler[0], good_euler[1], good_euler[2]])
		badPose = np.array([bad_pos[0], bad_pos[1], bad_pos[2], bad_euler[0], bad_euler[1], bad_euler[2]])
		return goodPose

	def get_fk(self):
		# Synchronous FK via MoveIt2.compute_fk()
		js = JointState()
		js.name = list(self.moveit2.joint_names)
		js.position = list(self._last_joint_msg)
		pose_stamped = self.moveit2.compute_fk(
			joint_state=js,
			fk_link_names=[self.end_effector_name],
		)
		if pose_stamped is None:
			self.get_logger().warn("FK failed or returned empty result")
			return
		if isinstance(pose_stamped, list):
			pose_stamped = pose_stamped[0] if pose_stamped else None
			if pose_stamped is None:
				self.get_logger().warn("FK returned empty list")
				return

		p = pose_stamped.pose.position
		q = pose_stamped.pose.orientation
		# Compute relative orientation to hardcoded home quaternion so home is (0,0,0)
		qx, qy, qz, qw = q.x, q.y, q.z, q.w
		
		roll, pitch, yaw = quat_to_euler(qx, qy, qz, qw)
		frame = pose_stamped.header.frame_id or self.base_link_name

		self.pose = np.array([p.x, p.y, p.z, roll, pitch, yaw])
		self.quat = np.array([qx, qy, qz, qw])
		self.frame = frame
		#print("Computed fk (sync)")


	def _on_timer(self):
		# Ensure joint states have been received at least once
		if not self._last_joint_msg:
			self.get_logger().warn("Waiting for joint_states...")
			return
		# Always update pose
		self.pose = self.get_frame()
		# Only print if enabled
		if self.enable_pose_print:
			print(
				f"[GripperPose] frame={self.frame} pos=({self.pose[0]:.4f}, {self.pose[1]:.4f}, {self.pose[2]:.4f}) "
				f"quat=({self.quat[0]:.4f}, {self.quat[1]:.4f}, {self.quat[2]:.4f}, {self.quat[3]:.4f}) "
				f"rpy=({self.pose[3]:.4f}, {self.pose[4]:.4f}, {self.pose[5]:.4f})"
			)


def main(argv=None):
	rclpy.init(args=argv)
	node = PoseReader()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main(sys.argv)
