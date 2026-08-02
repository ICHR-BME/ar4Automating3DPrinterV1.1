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
from rclpy.duration import Duration

def quat_to_euler(x: float, y: float, z: float, w: float):
	roll, pitch, yaw = R.from_quat([x, y, z, w]).as_euler("XYZ", degrees=False)
	return roll, pitch, yaw

# moveit_msgs error codes -> readable names
_MOVEIT_ERR = {
	1: "SUCCESS", 99999: "FAILURE",
	-1: "PLANNING_FAILED", -2: "INVALID_MOTION_PLAN",
	-3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE", -4: "CONTROL_FAILED",
	-5: "UNABLE_TO_AQUIRE_SENSOR_DATA", -6: "TIMED_OUT", -7: "PREEMPTED",
	-10: "START_STATE_IN_COLLISION", -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
	-12: "GOAL_IN_COLLISION", -13: "GOAL_VIOLATES_PATH_CONSTRAINTS", -14: "GOAL_CONSTRAINTS_VIOLATED",
	-15: "INVALID_GROUP_NAME", -16: "INVALID_GOAL_CONSTRAINTS", -17: "INVALID_ROBOT_STATE",
	-18: "INVALID_LINK_NAME", -19: "INVALID_OBJECT_NAME",
	-21: "FRAME_TRANSFORM_FAILURE", -22: "COLLISION_CHECKING_UNAVAILABLE",
	-23: "ROBOT_STATE_STALE", -24: "SENSOR_INFO_STALE", -25: "COMMUNICATION_FAILURE",
	-31: "NO_IK_SOLUTION",
}

def _moveit_err_str(moveit2):
	"""Readable error string for the last MoveIt execution, or '?'."""
	try:
		code = moveit2.get_last_execution_error_code()
		val = getattr(code, "val", code)
		return f"{_MOVEIT_ERR.get(val, 'UNKNOWN')}({val})"
	except Exception:
		return "?"

# patched local copy of pymoveit2's moveit2.py
from .moveit2 import MoveIt2

class PoseReader(Node):
	"""Node that tracks (and optionally prints) the gripper pose via pymoveit2."""

	def __init__(self, node_name: Optional[str] = None, enable_pose_print: bool = True,
	             robot: str = 'ar4'):
		super().__init__(node_name or "gripper_pose_reader")

		from .robot_config import get_robot_config
		self.robot = robot
		self.robot_config = get_robot_config(robot)
		joint_names = self.robot_config['joint_names']
		base_link_name = self.robot_config['base_link']
		end_effector_name = self.robot_config['end_effector_link']
		group_name = self.robot_config['move_group']

		self._cb_group = ReentrantCallbackGroup()
		self.moveit2 = MoveIt2(
			node=self,
			joint_names=joint_names,
			base_link_name=base_link_name,
			end_effector_name=end_effector_name,
			group_name=group_name,
			use_move_group_action=True,
			callback_group=self._cb_group,
		)

		# 0.0 makes MoveIt warn and fall back to 1.0
		self.moveit2.max_velocity = 0.9
		self.moveit2.max_acceleration = 0.9

		# pause after each move so TrajectoryExecutionManager releases before the next command
		self.move_settle_delay = 0.5


		self.base_link_name = base_link_name
		self.end_effector_name = end_effector_name
		# where a grasped object hangs: the gripper's TCP frame when the URDF
		# has one, else the eef flange (see robot_config 'grasp_frame')
		self.grasp_frame_name = self.robot_config.get(
			'grasp_frame', end_effector_name)

		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

		self._last_joint_msg = None  # list[float] ordered by self.moveit2.joint_names
		self.joint_state_by_name = {}  # every joint by name, gripper included
		self._fk_future = None
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
		self._timer = self.create_timer(0.5, self._on_timer)

		self.get_logger().info(
			f"PoseReader started; base='{base_link_name}', eef='{end_effector_name}'"
		)
		# rotation from Bad Frame to Good Frame + neutral tool euler offset
		self.frameRotationAngles = self.robot_config['frame_rotation_angles']
		self.frameOffsetAngles = self.robot_config['frame_offset_angles']

		# publishes 'stop' to kill any in-flight trajectory before sending a new goal
		self._cancellation_pub = self.create_publisher(String, '/trajectory_execution_event', 1)

	def _cancel_and_wait(self, wait_timeout=3.0, grace=0.75):
		"""Cancel any in-flight MoveIt trajectory and wait for it to go idle.

		The 'stop' is only published if a motion still looks in-flight after a
		short grace period. An unconditional stop races the natural end of the
		previous move (ground truth trips ~settle_delay before the controller
		finishes), and a cancel landing at the exact moment of completion
		wedges move_group's TrajectoryExecutionManager: the stop handler
		blocks joining the execution thread while holding
		execution_thread_mutex_, after which every new goal plans but never
		executes until move_group is restarted."""
		def _busy():
			return (getattr(self.moveit2, '_MoveIt2__is_executing', False) or
			        getattr(self.moveit2, '_MoveIt2__is_motion_requested', False))

		# grace: let a just-finished move deliver its result instead of
		# cancelling it mid-completion
		_deadline = time.time() + grace
		while _busy() and time.time() < _deadline:
			time.sleep(0.05)

		if _busy():
			_stop = String()
			_stop.data = 'stop'
			self._cancellation_pub.publish(_stop)
			# let the result callback clear __is_executing before the next goal arrives
			_deadline = time.time() + wait_timeout
			while _busy():
				if time.time() > _deadline:
					break
				time.sleep(0.05)
		# force-clear in case the server never sent a result (hard timeout, restart)
		self.moveit2._MoveIt2__is_motion_requested = False
		self.moveit2._MoveIt2__is_executing = False
		# give TrajectoryExecutionManager a beat to release the lock
		time.sleep(0.3)

	def _reached_configuration(self, joint_positions, tol=0.10):
		"""True if measured joints are within tol rad of target.

		Ground-truth completion check, independent of pymoveit2's result callback.
		tol=0.10 rad matches the controller's loosest goal tolerance (J6)."""
		actual = self._last_joint_msg
		if actual is None or len(actual) != len(joint_positions):
			return False
		return all(abs(a - t) <= tol for a, t in zip(actual, joint_positions))

	def _eef_pose_truth(self):
		"""link_6 pose in base_link as (pos, quat_xyzw), or (None, None).
		TF first, FK fallback: a starved TF buffer would fake "not reached"
		and cause false timeouts and retry thrash."""
		try:
			tf = self.tf_buffer.lookup_transform(
				self.base_link_name, self.end_effector_name, Time(),
				timeout=Duration(seconds=0.1))
			t = tf.transform.translation
			r = tf.transform.rotation
			return (np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w]))
		except Exception:
			pass
		if not self._last_joint_msg:
			return (None, None)
		try:
			js = JointState()
			js.name = list(self.moveit2.joint_names)
			js.position = list(self._last_joint_msg)
			ps = self.moveit2.compute_fk(
				joint_state=js, fk_link_names=[self.end_effector_name])
		except Exception:
			return (None, None)
		if isinstance(ps, list):
			ps = ps[0] if ps else None
		if ps is None:
			return (None, None)
		# Only trust FK expressed in the base frame we compare against.
		frame = (ps.header.frame_id or self.base_link_name).lstrip('/')
		if frame != self.base_link_name.lstrip('/'):
			return (None, None)
		p = ps.pose.position
		q = ps.pose.orientation
		return (np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w]))

	def _reached_pose(self, target_pos, target_quat, pos_tol=0.025, ang_tol=0.17):
		"""Ground-truth check that link_6 is within tolerance of the target.
		Deliberately loose (~2.5 cm / ~10 deg): the joint goal tolerances leave
		that much slack on a settled move, tighter would false-fail."""
		cur_pos, cur_q = self._eef_pose_truth()
		if cur_pos is None:
			return False
		if np.linalg.norm(cur_pos - np.asarray(target_pos, dtype=float)) > pos_tol:
			return False
		tq = np.asarray(target_quat, dtype=float)
		# Angle between orientations; |dot| handles the q/-q double cover.
		dot = min(1.0, abs(float(np.dot(cur_q, tq))))
		return (2.0 * np.arccos(dot)) <= ang_tol

	def move_to_configuration(self, joint_positions, timeout=15.0, max_retries=2):
		"""Joint-space move with retries. Polls with a deadline instead of
		wait_until_executed (which has none), and counts the move done once the
		arm physically reaches the config, so a missed result callback can't
		turn a completed move into a timeout."""
		for attempt in range(max_retries + 1):
			if attempt > 0:
				# a "failed" attempt may actually have arrived (missed result
				# callback); cancelling a completed move is what desyncs the arm
				if self._reached_configuration(joint_positions):
					time.sleep(self.move_settle_delay)
					return True
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
				# ground truth: arm reached the config. A missed result callback
				# leaves the flags stuck set, which would read as a false timeout.
				if self._reached_configuration(joint_positions):
					time.sleep(self.move_settle_delay)
					return True
				if time.time() > _deadline:
					self.get_logger().error(
						f"[move_to_configuration] timed out after {timeout}s."
					)
					timed_out = True
					self._cancel_and_wait()
					break
				time.sleep(0.05)

			# only ground truth counts; motion_suceeded is racy both ways
			if self._reached_configuration(joint_positions):
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

	def execute_joint_trajectory(self, joint_trajectory, timeout_margin=10.0):
		"""Run a prebuilt JointTrajectory as ONE continuous motion via
		move_group's ExecuteTrajectory action (no per-point replanning or
		stops). Completion is judged like move_to_configuration: pymoveit2's
		flags polled with a deadline, plus the ground-truth joint check
		against the trajectory's final point. No retries — a partially run
		trajectory leaves the arm mid-path, so the caller decides recovery."""
		if not joint_trajectory.points:
			return False
		final_config = list(joint_trajectory.points[-1].positions)
		last_t = joint_trajectory.points[-1].time_from_start
		timeout = last_t.sec + last_t.nanosec * 1e-9 + timeout_margin

		self._cancel_and_wait()
		self.moveit2.motion_suceeded = False
		self.moveit2.execute(joint_trajectory)

		_deadline = time.time() + timeout
		while (getattr(self.moveit2, '_MoveIt2__is_motion_requested', False) or
		       getattr(self.moveit2, '_MoveIt2__is_executing', False)):
			if time.time() > _deadline:
				self.get_logger().error(
					f"[execute_joint_trajectory] timed out after {timeout:.1f}s."
				)
				self._cancel_and_wait()
				break
			time.sleep(0.05)

		# only ground truth counts (see move_to_configuration)
		if self._reached_configuration(final_config):
			time.sleep(self.move_settle_delay)
			return True
		err = _moveit_err_str(self.moveit2)
		self.get_logger().error(
			f"[execute_joint_trajectory] did not reach the final config (MoveIt err={err})."
		)
		return False

	def _plan_linear_trajectory(self, bad_pos, q_msg, plan_timeout=5.0, max_step=0.0025):
		"""Cartesian straight-line plan to the target (base_link frame).
		Returns the planned JointTrajectory, or None unless the FULL line is
		feasible — a partial scrape stroke is worse than no move. Uses
		plan_async and polls the future (the background executor completes
		it); the vendored blocking plan() would spin_once a node it doesn't
		own."""
		future = self.moveit2.plan_async(
			position=Point(x=float(bad_pos[0]), y=float(bad_pos[1]), z=float(bad_pos[2])),
			quat_xyzw=q_msg,
			cartesian=True,
			max_step=max_step,
		)
		if future is None:
			return None
		_deadline = time.time() + plan_timeout
		while not future.done():
			if time.time() > _deadline:
				self.get_logger().warn(
					"[move_to_pose] Cartesian path service did not answer in "
					f"{plan_timeout:.1f}s."
				)
				return None
			time.sleep(0.02)
		# 0.999 = the whole line, without an exact float compare against 1.0
		return self.moveit2.get_trajectory(
			future, cartesian=True, cartesian_fraction_threshold=0.999
		)

	def move_to_pose(self, pos, euler, max_retries=2, linear=False):
		"""Pose move with retries; cancels any in-flight trajectory before
		each attempt so MoveIt starts from a clean idle state. linear=True
		plans a straight Cartesian line to the target and refuses to move
		unless the entire line is feasible."""
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
				# the previous attempt may actually have arrived; cancelling
				# a completed move desyncs the arm
				if self._reached_pose(bad_pos, q):
					time.sleep(self.move_settle_delay)
					return True
				self.get_logger().warn(
					f"[move_to_pose] Retry {attempt}/{max_retries}…"
				)

			# wait for idle before sending a new goal, else it aborts immediately
			self._cancel_and_wait()
			# reset so a stale result can't count for this attempt
			self.moveit2.motion_suceeded = False

			attempt_timeout = _timeout
			if linear:
				jt = self._plan_linear_trajectory(bad_pos, q_msg)
				if jt is None or not jt.points:
					if attempt < max_retries:
						self.get_logger().warn(
							f"[move_to_pose] linear (Cartesian) planning failed on "
							f"attempt {attempt + 1} — retrying…"
						)
						continue
					self.get_logger().error(
						"[move_to_pose] linear (Cartesian) planning failed — "
						"all attempts exhausted."
					)
					return False
				last_t = jt.points[-1].time_from_start
				attempt_timeout = max(_timeout, last_t.sec + last_t.nanosec * 1e-9 + 5.0)
				self.moveit2.execute(jt)
			else:
				self.moveit2.move_to_pose(
					position=Point(x=bad_pos[0], y=bad_pos[1], z=bad_pos[2]),
					quat_xyzw=q_msg,
				)

			# poll both flags; wait_until_executed can return early under the
			# multithreaded executor
			_deadline = time.time() + attempt_timeout
			timed_out = False
			while (getattr(self.moveit2, '_MoveIt2__is_motion_requested', False) or
			       getattr(self.moveit2, '_MoveIt2__is_executing', False)):
				# ground truth, same as move_to_configuration
				if self._reached_pose(bad_pos, q):
					time.sleep(self.move_settle_delay)
					return True
				if time.time() > _deadline:
					self.get_logger().error(
						f"[move_to_pose] timed out after {attempt_timeout}s waiting for trajectory to finish."
					)
					timed_out = True
					break
				time.sleep(0.05)

			# only ground truth counts (motion_suceeded is racy both ways)
			if self._reached_pose(bad_pos, q):
				time.sleep(self.move_settle_delay)
				return True

			reason = f"timed out after {attempt_timeout}s" if timed_out else "motion aborted/failed"
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
		"""Inverse of to_good_frame."""
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

		# every joint by name, including ones outside the planning group (the
		# gripper's drive_joint) — printer_automation watches it to tell when a
		# gripper trajectory finished or stalled against an object
		self.joint_state_by_name = dict(zip(msg.name, msg.position))

		try:
			self._last_joint_msg = [
				float(msg.position[msg.name.index(j)]) for j in self.moveit2.joint_names
			]
		except ValueError:
			# Missing joints in this message; skip
			return
	
	def get_frame(self, frame=None):
		frame = frame or self.end_effector_name
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
