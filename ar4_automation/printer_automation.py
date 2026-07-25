import warnings
# scipy's gimbal lock warning is benign (rotation fine, euler decomposition
# non-unique at +/-90). filter here so every entry script inherits it.
warnings.filterwarnings("ignore", message="Gimbal lock detected", category=UserWarning)

from .aruco_detector import ArucoDetectionViewer
import rclpy
from rclpy.time import Time
from rclpy.duration import Duration
import numpy as np
import os
import json
import csv
import datetime
import functools
from pymoveit2 import GripperInterface
from .printerclass import BambuPrinter
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf2_ros
from .simulated3DPrinter import Simulated3DPrinter
import time
import threading


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_DATA_DIR = os.path.join(_REPO_ROOT, "data")
_LOG_DIR = os.path.join(_DATA_DIR, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)


class MarkerNotVisibleError(RuntimeError):
    """Raised when a scan reaches its viewing pose but the target marker is
    not detected. Left uncaught it terminates the calling procedure/script."""


def _timed(method):
    """Log wall-clock duration + call chain for a public method."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        call_chain = " > ".join(self._timing_call_stack + [method.__name__])
        self._timing_call_stack.append(method.__name__)
        paused_at_start = self._timing_total_paused
        t0 = time.perf_counter()
        try:
            result = method(self, *args, **kwargs)
        finally:
            elapsed = round((time.perf_counter() - t0) - (self._timing_total_paused - paused_at_start), 4)
            self._timing_call_stack.pop()
            self._record_timing(call_chain, elapsed)
        return result
    return wrapper


class printerAutomation(ArucoDetectionViewer):
    def __init__(self, calibration_mode=False, stream_source="webcam", camera_index=None, camera_keyword="GENERAL WEBCAM",
                 color_topic=None, depth_topic=None, camera_info_topic=None,
                 feed_rotation_deg=0.0, marker_sizes=None, robot='ar4'):
        self._startup_start = time.perf_counter()
        # camera topic defaults resolve from the robot config in ArucoDetectionViewer
        super().__init__(source=stream_source,
                         camera_index=camera_index,
                         camera_keyword=camera_keyword,
                         color_topic=color_topic,
                         depth_topic=depth_topic,
                         camera_info_topic=camera_info_topic,
                         feed_rotation_deg=feed_rotation_deg,
                         marker_sizes=marker_sizes,
                         calibration_file=os.path.join(_REPO_ROOT, "calibration", "camera_matrix.npz"),
                         robot=robot)
        self.get_logger().info(f"printerAutomation initialized, robot={robot}, calibration_mode={calibration_mode}")

        self.estimatedMarkerPrefix = "estimated_marker_"

        # gripper no-op switch, sim workaround
        self.gripper_disabled = False
        # add noise to estimated markers, for scan robustness testing
        self.randomize_estimated_markers = False
        # 10s observation window instead of 1s, for noise data collection
        self.collect_orientation_noise_data = False
        # extra scan passes re-aim at the fresh detection for a head-on measurement
        self.scan_passes = 1
        # scans travel straight to the estimated viewing pose, observe, then
        # follow any CORRECTION of that pose in steps of at most this (m),
        # re-observing between steps — so a poor initial estimate is walked in
        # gradually instead of chased across the workspace in one big move
        self.scan_approach_max_step = 0.50
        # recorded {'traj': file} guide paths are downsampled to joint configs
        # at most this far apart (rad, max over any joint) before replaying —
        # only used by the per-config fallback walk
        self.traj_guide_step = 0.35
        # preferred replay: the whole recording as ONE continuous trajectory.
        # speed multiplies the demonstrated pace; point_dt is the spacing (s,
        # in demonstrated time) of the points sent to the controller
        self.traj_guide_speed = 1.0
        self.traj_guide_point_dt = 0.10
        self._traj_guide_cache = {}
        self._traj_samples_cache = {}

        # Offset configs: per printer type, one waypoint list per procedure
        # (pickup/place/scrape) in the marker's local frame. Each procedure
        # runs exactly its own list, so every action is visible here.
        # Entry kinds:
        #   {'pos': [x,y,z], 'angle_deg': tilt about marker X (None = untilted)}
        #       add 'linear': True to force a straight Cartesian line to that
        #       waypoint (scrape strokes); the move fails rather than run a
        #       partial or non-straight path
        #   {'scan': viewing_distance_m}   step-limited approach + must see the
        #                                  marker; retried closer if the move fails
        #   {'move': viewing_distance_m}   direct move to the viewing pose at that
        #                                  distance — no step limit, no detection
        #   {'gripper': 'open'|'close'}    close records joints for the replay
        #   {'rotate': degrees}            roll the wrist by -degrees and back
        #                                  (dislodge debris); failure only warns
        #   {'traj': 'traj/<file>.traj'}   replay a recorded joint path (rough
        #                                  collision-free guide between phases,
        #                                  e.g. pickup end -> scrape approach);
        #                                  robot-specific joint space!
        # descriptions are for humans only. Keep pre-grasp and grasp aligned in
        # marker X/Y so the grasp move is pure marker Z.
        ''''pickup': [
                                        
                                        {'description': 'open gripper for the approach',
                                            'gripper': 'open'},
                                        {'description': 'scan the marker before approaching',
                                            'scan': 0.15},
                                        {'description': 'scan the marker before approaching',
                                            'scan': 0.125},
                                        {'description': 'approach standoff in front of the handle',
                                            'pos': np.array([0.0, 0.065, 0.125]), 'angle_deg': 0.0},
                                        {'description': 'grasp pose at the handle',
                                            'pos': np.array([0.0, 0.065, 0.082]), 'angle_deg': 0.0},
                                        {'description': 'grab the handle',
                                            'gripper': 'close'},
                                        {'description': 'grasp pose at the handle',
                                            'pos': np.array([0.0, 0.065, 0.082]), 'angle_deg': -5.0},
                                        {'description': 'lift / carry pose',
                                            'pos': np.array([0.0, 0.20, 0.082]), 'angle_deg': 10.0},
                                    ],'''
        self.offset_configs = {
            'printer_offset_davis': {
                'pickup': [
                    {'description': 'scan the marker before approaching',
                     'scan': 0.15},
                    {'description': 'scan the marker before approaching',
                     'scan': 0.15},
                    {'description': 'open gripper for the approach',
                     'gripper': 'open'},
                    {'description': 'approach standoff in front of the handle',
                     'pos': np.array([0.0, 0.06, 0.15]), 'angle_deg': 0.0},
                    {'description': 'grasp pose at the handle',
                     'pos': np.array([0.0, 0.06, 0.1]), 'angle_deg': 0.0},
                    {'description': 'grab the handle',
                     'gripper': 'close'},
                    {'description': 'lift / carry pose',
                     'pos': np.array([0.0, 0.14, 0.11]), 'angle_deg': 0.0},
                ],
                'place': [
                    {'description': 'lift / carry pose',
                     'pos': np.array([0.0, 0.14, 0.11]), 'angle_deg': -10.0},
                    {'description': 'partial descent, tucked toward the marker',
                     'pos': np.array([0.0, 0.105, 0.11]), 'angle_deg': -10.0},
                    {'description': 'shift out along marker Z',
                     'pos': np.array([0.0, 0.105, 0.09]), 'angle_deg': -10.0},
                    {'description': 'set down',
                     'pos': np.array([0.0, 0.06, 0.09]), 'angle_deg': -10.0},
                    {'description': 'release the handle',
                     'gripper': 'open'},
                    {'description': 'withdraw to the approach standoff',
                     'pos': np.array([0.0, 0.06, 0.15]), 'angle_deg': 0.0},
                ],
                'scrape': None,
            },
            
            'printer_offset': {
                'pickup': [
                    
                    {'description': 'open gripper for the approach',
                        'gripper': 'open'},
                    {'description': 'scan the marker before approaching',
                        'move': 0.20},
                    {'description': 'scan the marker before approaching',
                        'scan': 0.20},
                    {'description': 'scan the marker before approaching',
                        'scan': 0.125},
                    {'description': 'approach standoff in front of the handle',
                        'pos': np.array([0.0, 0.075, 0.125]), 'angle_deg': 0.0},
                    {'description': 'grasp pose at the handle',
                        'pos': np.array([0.0, 0.075, 0.08]), 'angle_deg': 0.0},
                    {'description': 'grab the handle',
                        'gripper': 'close'},
                    {'description': 'grasp pose at the handle',
                        'pos': np.array([0.0, 0.075, 0.08]), 'angle_deg': 0.0},
                    {'description': 'lift / carry pose',
                       'pos': np.array([0.0, 0.35, 0.12]), 'angle_deg': -15.0},
                    #{'description': 'lift / carry pose',
                    #    'pos': np.array([0.0, 0.39, 0.20]), 'angle_deg': -90.0},
                ],
                'place': [
                    #{'description': 'lift / carry pose',
                    #    'pos': np.array([0.0, 0.40, 0.20]), 'angle_deg': -30.0},
                    {'description': 'lift / carry pose',
                        'pos': np.array([0.0, 0.35, 0.10]), 'angle_deg': 0.0},
                    {'description': 'lift / carry pose',
                        'pos': np.array([0.0, 0.18, 0.10]), 'angle_deg': 0.0},
                    {'description': 'partial descent, tucked toward the marker',
                        'pos': np.array([0.0, 0.145, 0.10]), 'angle_deg': 0.0},
                    {'description': 'shift out along marker Z',
                        'pos': np.array([0.0, 0.145, 0.07]), 'angle_deg': 0.0},
                    {'description': 'set down',
                        'pos': np.array([0.0, 0.085, 0.07]), 'angle_deg': 0.0},
                    {'description': 'release the handle',
                        'gripper': 'open'},
                    {'description': 'withdraw to the approach standoff',
                        'pos': np.array([0.0, 0.085, 0.15]), 'angle_deg': 0.0},
                ],
                'scrape': None,
            },
            # Printer with the marker to the side; also the scrape fixture
            'box_offset_davis': {
                'pickup': [
                    {'description': 'scan the marker before approaching',
                     'scan': 0.15},
                    {'description': 'open gripper for the approach',
                     'gripper': 'open'},
                    {'description': 'approach standoff in front of the handle',
                     'pos': np.array([0.0, 0.03, 0.15]), 'angle_deg': None},
                    {'description': 'grasp pose at the handle',
                     'pos': np.array([0.0, 0.03, 0.102]), 'angle_deg': None},
                    {'description': 'grab the handle',
                     'gripper': 'close'},
                    {'description': 'lift / carry pose',
                     'pos': np.array([0.0, 0.13, 0.102]), 'angle_deg': None},
                ],
                'place': [
                    {'description': 'lift / carry pose',
                     'pos': np.array([0.0, 0.13, 0.102]), 'angle_deg': None},
                    {'description': 'descend back to the grasp pose',
                     'pos': np.array([0.0, 0.03, 0.102]), 'angle_deg': None},
                    {'description': 'release the handle',
                     'gripper': 'open'},
                    {'description': 'withdraw to the approach standoff',
                     'pos': np.array([0.0, 0.03, 0.15]), 'angle_deg': None},
                ],
                'scrape': [
                    {'description': 'scrape standoff along marker Z',
                     'pos': np.array([0.0, 0.092, 0.29]), 'angle_deg': 0.0},
                    {'description': 'full scrape depth',
                     'pos': np.array([0.0, 0.092, 0.13]), 'angle_deg': 0.0,
                     'linear': True},
                    {'description': 'retract to standoff',
                     'pos': np.array([0.0, 0.092, 0.29]), 'angle_deg': 0.0,
                     'linear': True},
                    #{'description': 'roll the wrist to dislodge debris',
                    # 'rotate': 60.0},
                ],
            },
            'box_offset': {
                'pickup': [
                    {'description': 'open gripper for the approach',
                        'gripper': 'open'},
                    {'description': 'scan the marker before approaching',
                        'scan': 0.15},
                    {'description': 'scan the marker before approaching',
                        'scan': 0.15},
                    {'description': 'scan the marker before approaching',
                        'scan': 0.125},
                    {'description': 'scan the marker before approaching',
                        'scan': 0.125},
                    {'description': 'approach standoff in front of the handle',
                        'pos': np.array([0.0, 0.06, 0.125]), 'angle_deg': 0.0},
                    {'description': 'grasp pose at the handle',
                        'pos': np.array([0.0, 0.06, 0.1]), 'angle_deg': 0.0},
                        {'description': 'grasp pose at the handle',
                        'pos': np.array([0.0, 0.06, 0.067]), 'angle_deg': 0.0},
                    {'description': 'grab the handle',
                        'gripper': 'close'},
                    '''{'description': 'grasp pose at the handle',
                        'pos': np.array([0.0, 0.05, 0.1]), 'angle_deg': -5.0},
                    {'description': 'lift / carry pose',
                        'pos': np.array([0.0, 0.14, 0.1]), 'angle_deg': 0.0},'''
                ],
                'place': [
                    {'description': 'lift / carry pose',
                        'pos': np.array([0.0, 0.13, 0.102]), 'angle_deg': None},
                    {'description': 'descend back to the grasp pose',
                        'pos': np.array([0.0, 0.03, 0.102]), 'angle_deg': None},
                    {'description': 'release the handle',
                        'gripper': 'open'},
                    {'description': 'withdraw to the approach standoff',
                        'pos': np.array([0.0, 0.03, 0.15]), 'angle_deg': None},
                ],
                'scrape': [
                    # lite6 recording: transit from the pickup end pose to the
                    # scrape approach along a demonstrated collision-free route
                    #{'description': 'recorded guide path from pickup to scrape approach',
                    #    'traj': 'traj/scrapeWaypoints3.traj'},
                    {'description': 'scrape standoff along marker Z',
                        'pos': np.array([0.0, -0.04, 0.40]), 'angle_deg': 5.0},
                    {'description': 'full scrape depth',
                        'pos': np.array([0.0, -0.04, 0.08]), 'angle_deg': 5.0,
                        'linear': True},
                    {'description': 'retract to standoff',
                        'pos': np.array([0.0, -0.04, 0.40]), 'angle_deg': 5.0,
                        'linear': True},
                    #{'description': 'roll the wrist to dislodge debris',
                     #   'rotate': 60.0},
                ],
            },
        }
        ## marker_id -> config name, unlisted ids fall back to 'box_offset'
        self.marker_offset_config = {}

        # tool orientation used to face a marker, and the camera's vertical
        # mount offset (m, base Z), both per robot
        self.offsetOri = self.robot_config['offset_ori']
        self.camera_z_offset = self.robot_config['camera_z_offset']

        # optional in-session correction (EEF frame, metres) to the eef->camera
        # translation, produced by calibrate_camera_offset(apply=True). None =
        # trust the URDF/TF mount as-is. Applied to BOTH the scan-targeting
        # backoff and the measured marker-in-base pose so a pickup uses it.
        self._camera_offset_correction = None
        # per-detection sample buffer for calibrate_camera_offset
        self._calib_collect_id = None
        self._calib_buf = []

        # state file, saved every 5s, loaded at startup
        self._state_save_path = os.path.join(_DATA_DIR, "printer_state.json")
        self.create_timer(5.0, self._auto_save_state)

        # timing log, one row per public-method call
        _timing_dir = os.path.join(_DATA_DIR, "timing")
        os.makedirs(_timing_dir, exist_ok=True)
        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._timing_csv_path = os.path.join(_timing_dir, f"timing_{_ts}.csv")
        self._timing_file = open(self._timing_csv_path, "w", newline="")
        self._timing_writer = csv.writer(self._timing_file)
        self._timing_call_stack = []
        self._timing_total_paused = 0.0
        self._timing_pause_start = None
        self._timing_writer.writerow(["timestamp", "call_chain", "duration_s"])
        self._timing_file.flush()

        # raw scan log, one row per frame, truncated on restart
        self._scan_log_path = os.path.join(_LOG_DIR, "scan_raw_measurements.csv")
        self._scan_log_marker_id = None   # set by scanToMarker while active
        self._scan_log_distance = None
        self._scan_log_movement_id = 0     # bumped per observation window
        self._scan_log_file = open(self._scan_log_path, 'w', newline='')
        self._scan_log_writer = csv.writer(self._scan_log_file)
        self._scan_log_writer.writerow([
            'marker_id', 'scan_distance', 'movement_id',
            'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw',
            'cam_px', 'cam_py', 'cam_pz', 'cam_qx', 'cam_qy', 'cam_qz', 'cam_qw',
        ])
        self._scan_log_file.flush()

        # marker_id -> BambuPrinter, filled by register_bambu_printer()
        self._bambu_printers: dict = {}

        # recorded pickup joints ({'marker','grasp','lift'}) so placePlate can
        # replay them instead of using flip-prone pose IK
        self._pickup_replay = None

        # Gripper interface (robots without one configured run gripper-disabled).
        # Two kinds: 'moveit_action' drives pymoveit2's GripperInterface (AR4);
        # 'lite6_service' calls the xarm driver's open/close services (Lite 6).
        gripper_cfg = self.robot_config['gripper']
        self.gripper = None
        self._gripper_kind = None
        self._gripper_open_client = None
        self._gripper_close_client = None
        if gripper_cfg is None:
            self.gripper_disabled = True
            self.get_logger().info(
                f"No gripper configured for robot '{robot}'; gripper commands are skipped."
            )
        else:
            self._gripper_kind = gripper_cfg.get('kind', 'moveit_action')
            if self._gripper_kind == 'lite6_service':
                from xarm_msgs.srv import Call
                self._gripper_open_client = self.create_client(
                    Call, gripper_cfg['open_service'], callback_group=self._cb_group)
                self._gripper_close_client = self.create_client(
                    Call, gripper_cfg['close_service'], callback_group=self._cb_group)
                self.get_logger().info(
                    f"Lite 6 gripper via services {gripper_cfg['open_service']} / "
                    f"{gripper_cfg['close_service']}."
                )
            else:
                self.gripper = GripperInterface(
                    node=self,
                    callback_group=self._cb_group,
                    **{k: v for k, v in gripper_cfg.items() if k != 'kind'},
                )

    def _record_timing(self, call_chain: str, duration_s: float):
        """Append one timing row to the session CSV."""
        self._timing_writer.writerow(
            [datetime.datetime.now().isoformat(), call_chain, duration_s]
        )
        self._timing_file.flush()

    def record_startup_time(self):
        """Log time from __init__ to now as a 'startup' timing row."""
        elapsed = round(time.perf_counter() - self._startup_start, 4)
        self._record_timing("startup", elapsed)

    def pause_timing(self):
        """Pause the timing clock; paused time is excluded from active timers."""
        if self._timing_pause_start is None:
            self._timing_pause_start = time.perf_counter()

    def resume_timing(self):
        """Resume the timing clock."""
        if self._timing_pause_start is not None:
            self._timing_total_paused += time.perf_counter() - self._timing_pause_start
            self._timing_pause_start = None

    # ---- State persistence ----

    def save_state(self):
        """Dump marker poses, offset config, and printer configs to JSON."""
        data = {
            "marker_offset_config": {str(k): v for k, v in self.marker_offset_config.items()},
            "markers": [],
            "printers": getattr(self, '_saved_printer_configs', []),
        }
        for entry in self.stream.found_markers.values():
            if 'positionInBase' not in entry or 'eulerInBase' not in entry:
                continue
            data["markers"].append({
                "id": int(entry['id']),
                "positionInBase": entry['positionInBase'].tolist(),
                "eulerInBase": entry['eulerInBase'].tolist(),
                "dict_name": entry.get('dict_name', 'unknown'),
                "marker_size": float(entry.get('marker_size', 0.024)),
                "estimated": bool(entry.get('estimated', False)),
            })
        try:
            with open(self._state_save_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"save_state: could not write {self._state_save_path}: {e}")

    def register_printers(self, printers):
        """Store printer configs (dicts with marker_id/pos/orient/door_marker_texture) so save_state includes them."""
        self._saved_printer_configs = [
            {
                "marker_id": int(p["marker_id"]),
                "pos": list(p["pos"]),
                "orient": list(p["orient"]),
                "door_marker_texture": p["door_marker_texture"],
            }
            for p in printers
        ]

    def load_state(self):
        """Restore marker poses and offset config from the save file; markers
        come back as estimates so the next scan overwrites them."""
        if not os.path.exists(self._state_save_path):
            self.get_logger().info(f"load_state: no save file at {self._state_save_path} — starting fresh")
            return False
        try:
            with open(self._state_save_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.get_logger().warn(f"load_state: could not read {self._state_save_path}: {e}")
            return False

        # json keys are strings
        for k, v in data.get("marker_offset_config", {}).items():
            self.marker_offset_config[int(k)] = v

        for m in data.get("markers", []):
            marker_id = int(m["id"])
            pos = np.array(m["positionInBase"], dtype=float)
            euler = np.array(m["eulerInBase"], dtype=float)
            self.register_estimated_marker(marker_id=marker_id, bad_pos=pos, bad_euler=euler)
            # keep real detections marked as real so they don't get replaced
            # by a geometric fallback later
            if not m.get("estimated", True):
                self.stream.found_markers[marker_id]['estimated'] = False

        if data.get("printers"):
            self._saved_printer_configs = data["printers"]

        n = len(data.get("markers", []))
        self.get_logger().info(
            f"load_state: restored {n} marker(s) and offset config from {self._state_save_path}"
        )
        return True

    def _auto_save_state(self):
        self.save_state()

    # ---- Raw measurement logging ----

    def _on_raw_marker_measurement(self, marker_id, pos_in_base, quat_in_base,
                                    pos_in_camera, quat_in_camera):
        """Per-frame raw detection hook; writes a CSV row immediately, but
        only while a scanToMarker observation window is active."""
        # feed the camera-offset calibration collector (raw, unfiltered base +
        # camera-frame positions; the latter shows WHICH camera axis mismatches)
        if self._calib_collect_id is not None and marker_id == self._calib_collect_id:
            self._calib_buf.append((np.array(pos_in_base, dtype=float),
                                    np.array(pos_in_camera, dtype=float)))
        if self._scan_log_marker_id is None:
            return
        if marker_id != self._scan_log_marker_id:
            return
        row = [
            marker_id,
            round(self._scan_log_distance, 6) if self._scan_log_distance is not None else '',
            self._scan_log_movement_id,
            round(float(pos_in_base[0]), 6),
            round(float(pos_in_base[1]), 6),
            round(float(pos_in_base[2]), 6),
            round(float(quat_in_base[0]), 6),
            round(float(quat_in_base[1]), 6),
            round(float(quat_in_base[2]), 6),
            round(float(quat_in_base[3]), 6),
            round(float(pos_in_camera[0]), 6),
            round(float(pos_in_camera[1]), 6),
            round(float(pos_in_camera[2]), 6),
            round(float(quat_in_camera[0]), 6),
            round(float(quat_in_camera[1]), 6),
            round(float(quat_in_camera[2]), 6),
            round(float(quat_in_camera[3]), 6),
        ]
        self._scan_log_writer.writerow(row)
        self._scan_log_file.flush()

    # ---- Helpers ----

    def _find_marker_entry(self, marker_id):
        """Look up a marker by ID from marker_poses. Returns entry dict or None."""
        for m in self.marker_poses:
            if m['id'] == marker_id:
                return m
        return None

    def _broadcast_static_tf(self, bad_pos, bad_euler, child_frame):
        """Broadcast a static TF for a marker pose in base_link."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.base_link_name
        t.child_frame_id = child_frame
        t.transform.translation.x = float(bad_pos[0])
        t.transform.translation.y = float(bad_pos[1])
        t.transform.translation.z = float(bad_pos[2])
        q = R.from_euler("XYZ", bad_euler, degrees=False).as_quat()
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])
        self.tf2_static_broadcaster.sendTransform(t)

    def _apply_offset_in_marker_frame(self, marker_pos, marker_euler, offset_pos, offset_ori):
        """Apply an offset in the marker's local frame, return (pos, euler) in base_link."""
        R_marker = R.from_euler("XYZ", marker_euler, degrees=False)
        target_pos = marker_pos + R_marker.apply(offset_pos)
        target_euler = (R_marker * R.from_euler("XYZ", offset_ori, degrees=False)).as_euler("XYZ", degrees=False)
        return target_pos, target_euler

    def _eef_to_camera_translation(self):
        """Translation of the camera frame origin expressed in the
        end-effector frame, from TF (the same camera link shown in RViz).
        Cached — the camera is rigidly bolted to the wrist, so it's constant.
        Returns None if TF isn't up yet (caller falls back to camera_z_offset)."""
        if getattr(self, '_t_eef_cam', None) is not None:
            return self._t_eef_cam
        camera_frame = self.camera_frame_name
        try:
            tf = self.tf_buffer.lookup_transform(
                self.end_effector_name, camera_frame,
                Time(), timeout=Duration(seconds=2.0))
        except Exception as e:
            self.get_logger().warn(
                f"_eef_to_camera_translation: TF {self.end_effector_name} -> "
                f"{camera_frame} unavailable ({e}); falling back to camera_z_offset."
            )
            return None
        t = tf.transform.translation
        self._t_eef_cam = np.array([t.x, t.y, t.z])
        self.get_logger().info(
            f"Camera offset from {self.end_effector_name}: "
            f"{np.round(self._t_eef_cam, 4)} m (frame {camera_frame})"
        )
        return self._t_eef_cam

    def _eef_to_camera_rotation(self):
        """Rotation R_(eef<-camera) from TF, cached (rigid mount). None if TF
        isn't up."""
        if getattr(self, '_R_eef_cam', None) is not None:
            return self._R_eef_cam
        try:
            tf = self.tf_buffer.lookup_transform(
                self.end_effector_name, self.camera_frame_name,
                Time(), timeout=Duration(seconds=2.0))
        except Exception:
            return None
        q = tf.transform.rotation
        self._R_eef_cam = R.from_quat([q.x, q.y, q.z, q.w])
        return self._R_eef_cam

    def _base_from_eef_transform(self):
        """(position, rotation) of the EEF in base_link from TF (latest), or
        (None, None) if unavailable."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_link_name, self.end_effector_name,
                Time(), timeout=Duration(seconds=0.2))
        except Exception:
            return None, None
        t = tf.transform.translation
        q = tf.transform.rotation
        return np.array([t.x, t.y, t.z]), R.from_quat([q.x, q.y, q.z, q.w])

    def _base_from_eef_rotation(self):
        """Rotation R_(base<-eef) from TF (latest available), or None."""
        _pos, rot = self._base_from_eef_transform()
        return rot

    def _enrich_marker_pose(self, entry):
        """Add the calibrated eef->camera translation correction to the measured
        marker-in-base pose. A translation error delta (EEF frame) in the camera
        mount biases the measurement by -R_(base<-eef) @ delta, so add it back."""
        entry = super()._enrich_marker_pose(entry)
        correction = getattr(self, '_camera_offset_correction', None)
        if entry is None or correction is None:
            return entry
        R_be = self._base_from_eef_rotation()
        if R_be is None:
            return entry
        entry['positionInBase'] = (np.asarray(entry['positionInBase'], dtype=float)
                                   + R_be.apply(correction))
        # keep the user-facing world position consistent (orientation unaffected)
        R_BF_GF = R.from_euler("XYZ", self.frameRotationAngles, degrees=False)
        entry['positionInWorld'] = R_BF_GF.apply(entry['positionInBase'])
        return entry

    def _camera_view_pose(self, marker_pos, marker_euler, offset_pos, offset_ori):
        """EEF pose (pos, euler in base_link) that lands the CAMERA frame at
        offset_pos/offset_ori in the marker frame — so the camera link (not the
        wrist) ends up facing the marker at the requested distance.

        offset_pos is the desired camera position in the marker frame (e.g.
        [0,0,viewing_distance] along the marker normal); offset_ori still sets
        the EEF orientation (the tuned per-robot self.offsetOri). The wrist is
        backed off the camera target by the fixed EEF->camera translation from
        TF, replacing the hand-tuned base-Z camera_z_offset shim. Falls back to
        that shim only if TF isn't available yet."""
        cam_pos, eef_euler = self._apply_offset_in_marker_frame(
            marker_pos, marker_euler, offset_pos, offset_ori)
        t_eef_cam = self._eef_to_camera_translation()
        if t_eef_cam is None:
            return cam_pos + np.array([0.0, 0.0, self.camera_z_offset]), eef_euler
        # fold in the calibrated correction (EEF frame) if one is active
        if self._camera_offset_correction is not None:
            t_eef_cam = t_eef_cam + self._camera_offset_correction
        # forward: cam_pos = eef_pos + R_eef @ t_eef_cam
        #  =>      eef_pos = cam_pos - R_eef @ t_eef_cam
        R_eef = R.from_euler("XYZ", eef_euler, degrees=False)
        eef_pos = cam_pos - R_eef.apply(t_eef_cam)
        return eef_pos, eef_euler

    def _tilted_offset_ori(self, angle_deg):
        """offsetOri tilted angle_deg about marker X (premultiplied, so the
        approach still runs along marker Z). None for zero angle."""
        if not angle_deg:
            return None
        tilt = R.from_euler("x", np.radians(angle_deg))
        return (tilt * R.from_euler("XYZ", self.offsetOri, degrees=False)).as_euler("XYZ", degrees=False)

    def _move_to_marker_offset(self, marker_id, offset_pos, offset_ori=None, linear=False):
        """Find the marker, apply the offset, move there. linear=True demands
        a straight Cartesian line to the target."""
        if offset_ori is None:
            offset_ori = self.offsetOri

        entry = self._find_marker_entry(marker_id)
        if entry is None:
            self.get_logger().warn(f"Marker ID {marker_id} not found in detected marker poses.")
            available_ids = [m['id'] for m in self.marker_poses]
            self.get_logger().info(f"Available marker IDs: {available_ids}")
            return False

        bad_pos = entry['positionInBase']
        bad_euler = entry['eulerInBase']

        badPos, badEuler = self._apply_offset_in_marker_frame(bad_pos, bad_euler, offset_pos, offset_ori)

        goodPos, goodEuler = self.to_good_frame(badPos, badEuler)
        self.get_logger().info(f'Moving to marker ID {marker_id} — marker centre: {bad_pos}, target: {badPos}')
        # markers are pinned by default, so the move can't drift any pose
        return self.move_to_pose(goodPos, goodEuler, linear=linear)

    def _get_waypoints_for_marker(self, marker_id, procedure):
        """Waypoint list for 'pickup'/'place'/'scrape', or None."""
        config_name = self.marker_offset_config.get(marker_id, 'box_offset')
        waypoints = self.offset_configs[config_name].get(procedure)
        return waypoints if waypoints else None

    def _follow_waypoints(self, markerID, waypoints, caller, tolerant_last=False):
        """Run the entries in order: moves, scans, gripper actions (a 'close'
        records joints for the pickup replay), wrist rotations. False on first
        failure, except tolerant_last lets a failed last position move (scrape
        retract) just warn."""
        self._walk_grasp_joints = None
        n = len(waypoints)
        # the last 'pos' move, not the last list entry — a rotate may follow it
        last_pos_i = max((i for i, wp in enumerate(waypoints) if 'pos' in wp),
                         default=None)
        for i, wp in enumerate(waypoints):
            if 'scan' in wp:
                # scanToMarker opens its own update window for markerID during
                # the observation; the marker stays pinned everywhere else
                if not self._scan_with_retries(markerID, float(wp['scan']), caller):
                    return False
                continue
            if 'move' in wp:
                # plain positioning at a viewing distance — no step limit, no
                # observation window, no detection requirement
                if not self.moveToViewingDistance(markerID, float(wp['move'])):
                    self.get_logger().error(
                        f"{caller}: move waypoint {i+1}/{n} "
                        f"(viewing distance {float(wp['move']):.3f} m) failed for marker {markerID}."
                    )
                    return False
                continue
            if 'gripper' in wp:
                if wp['gripper'] == 'close':
                    self._walk_grasp_joints = self._current_arm_joints()
                    self.close_gripper()
                    time.sleep(3.0)
                else:
                    self.open_gripper()
                continue
            if 'rotate' in wp:
                # wrist roll-and-return (e.g. dislodge debris after the scrape
                # retract); a failed rotation only warns — the walk continues
                self._rotate_wrist(float(wp['rotate']))
                continue
            if 'traj' in wp:
                # recorded rough path (joint space) guiding between phases
                # along a demonstrated collision-free route
                if not self._replay_traj_guide(wp['traj']):
                    self.get_logger().error(
                        f"{caller}: recorded guide path waypoint {i+1}/{n} failed "
                        f"for marker {markerID}."
                    )
                    return False
                continue
            tilt_ori = self._tilted_offset_ori(wp.get('angle_deg'))
            pos = np.asarray(wp['pos'], dtype=float)
            if not self._move_to_marker_offset(markerID, pos, tilt_ori,
                                               linear=bool(wp.get('linear'))):
                if tolerant_last and i == last_pos_i:
                    self.get_logger().error(
                        f"{caller}: last position waypoint {i+1}/{n} failed for marker {markerID}. Continuing."
                    )
                    return True
                self.get_logger().error(
                    f"{caller}: waypoint {i+1}/{n} failed for marker {markerID}."
                )
                return False
        return True

    # ---- BambuPrinter integration ----

    def register_bambu_printer(self, marker_id, printer: BambuPrinter):
        """Attach a connected BambuPrinter to a marker so transferPlate can
        move its head clear before pickup and home it after placing."""
        self._bambu_printers[marker_id] = printer
        self.get_logger().info(
            f"register_bambu_printer: marker {marker_id} → printer {printer.serial} at {printer.ip}"
        )
        printer.homing()

    # ---- Gripper ----

    def _call_gripper_service(self, client, label, timeout=5.0):
        """Call an empty-request xarm Call service (open/close_lite6_gripper)
        and wait on the result. Relies on the background executor to spin."""
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                f"{label} gripper: service {client.srv_name} unavailable "
                "(is the xarm driver running?)."
            )
            return False
        from xarm_msgs.srv import Call
        future = client.call_async(Call.Request())
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.02)
        if not future.done():
            self.get_logger().error(f"{label} gripper: service call timed out.")
            return False
        resp = future.result()
        if resp is not None and getattr(resp, 'ret', 0) != 0:
            self.get_logger().warn(
                f"{label} gripper: driver returned ret={resp.ret} "
                f"msg='{getattr(resp, 'message', '')}'."
            )
            return False
        return True

    def open_gripper(self):
        if self.gripper_disabled:
            self.get_logger().info("Gripper disabled — skipping open.")
            return
        self.get_logger().info("Opening gripper...")
        if self._gripper_kind == 'lite6_service':
            self._call_gripper_service(self._gripper_open_client, "open")
        else:
            self.gripper.open()

    def close_gripper(self):
        if self.gripper_disabled:
            self.get_logger().info("Gripper disabled — skipping close.")
            return
        self.get_logger().info("Closing gripper...")
        if self._gripper_kind == 'lite6_service':
            self._call_gripper_service(self._gripper_close_client, "close")
        else:
            self.gripper.close()

    # ---- Marker updates ----
    # Default-deny: every marker is pinned at all times, except inside an
    # explicit update window opened around a scan observation. This replaces
    # the old freeze/unfreeze (global pause during moves) and lock/unlock
    # (persistent per-marker pin) pair — both are the default now.

    def allow_marker_updates(self, marker_id=None):
        """Open a marker-update window: camera detections may update/add only
        marker_id — or ANY marker if None (location discovery). Sleeps first
        so frames captured while the robot was still moving drain before the
        window opens (camera pipeline lag)."""
        time.sleep(0.5)
        if marker_id is None:
            self.stream.allow_all_updates = True
            self.get_logger().info("Marker update window open for ALL markers.")
        else:
            self.stream.allowed_update_ids.add(marker_id)
            self.get_logger().info(f"Marker update window open for marker {marker_id}.")

    def block_marker_updates(self):
        """Close the update window — every marker is pinned again (the default)."""
        self.stream.allow_all_updates = False
        self.stream.allowed_update_ids.clear()
        self.get_logger().info("Marker update window closed — all markers pinned.")

    # ---- Marker registration & scanning ----

    def register_estimated_marker(self, marker_id, bad_pos, bad_euler):
        """Seed an estimated marker pose in TF and found_markers; the first
        real detection overwrites it. Writes directly (bypassing the camera
        update gate) — callers guard against clobbering real saved poses."""
        bad_pos = np.array(bad_pos, dtype=float)
        bad_euler = np.array(bad_euler, dtype=float)
        if self.randomize_estimated_markers:
            rng = np.random.default_rng()
            random_dir = rng.normal(size=3)
            random_dir /= np.linalg.norm(random_dir)
            bad_pos = bad_pos + random_dir * 0.03
            random_ori_dir = rng.normal(size=3)
            random_ori_dir /= np.linalg.norm(random_ori_dir)
            bad_euler = bad_euler + random_ori_dir * 0.05
        tf2Name = f"{self.markerNamePrefix}{marker_id}"

        self._broadcast_static_tf(bad_pos, bad_euler, tf2Name)

        # Compute good-frame values for display
        R_BF_GF = R.from_euler("XYZ", self.frameRotationAngles, degrees=False)
        goodPos = R_BF_GF.apply(bad_pos)
        goodEuler = (R_BF_GF * R.from_euler("XYZ", bad_euler, degrees=False)).as_euler("XYZ", degrees=False)

        entry = {
            'id': marker_id,
            'tf2Name': tf2Name,
            'positionInBase': bad_pos,
            'eulerInBase': bad_euler,
            'positionInWorld': goodPos,
            'orientInWorld': {
                'roll': np.degrees(goodEuler[0]),
                'pitch': np.degrees(goodEuler[1]),
                'yaw': np.degrees(goodEuler[2]),
            },
            'positionFromCamera': np.array([0.0, 0.0, 0.0]),
            'eulerFromCamera': np.array([0.0, 0.0, 0.0]),
            'orientFromCamera': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            'distanceFromCamera': 0.0,
            'estimated': True,
        }
        self.stream.found_markers[marker_id] = entry
        self.get_logger().info(
            f"Registered estimated marker {marker_id} at base_link pos={bad_pos}, euler={bad_euler}"
        )

    def _viewing_pose_for(self, marker_id, viewing_distance):
        """EEF pose (bad frame) that puts the camera viewing_distance out
        along marker_id's normal, from its current (possibly estimated) pose.
        None, None if the marker has no entry."""
        entry = self._find_marker_entry(marker_id)
        if entry is None:
            return None, None
        offset_pos = np.array([0.0, 0.0, viewing_distance])
        # the camera frame (from TF, not the wrist) lands at the offset — the
        # EEF is backed off by the fixed camera mount offset automatically
        return self._camera_view_pose(
            entry['positionInBase'], entry['eulerInBase'], offset_pos, self.offsetOri,
        )

    def _approach_viewing_pose(self, marker_id, viewing_distance):
        """Move to marker_id's viewing pose. The initial travel to the
        estimated pose is one direct, unclamped move; a short observation
        there lets the camera correct the estimate, and any resulting CHANGE
        of the viewing pose is then approached in steps of at most
        scan_approach_max_step (re-observing between steps) until the target
        stops moving. Returns the last move's success."""
        max_step = self.scan_approach_max_step
        # cap on correction steps; generous — reached only if the target keeps
        # receding, e.g. the estimate is refined away from the arm every step
        for step_i in range(15):
            target_pos, target_euler = self._viewing_pose_for(marker_id, viewing_distance)
            if target_pos is None:
                self.get_logger().error(
                    f"_approach_viewing_pose: marker {marker_id} has no pose entry."
                )
                return False
            eef_pos, _rot = self._base_from_eef_transform()
            remaining = None if eef_pos is None else float(np.linalg.norm(target_pos - eef_pos))
            if step_i > 0 and remaining is not None and remaining <= 0.01:
                # the observation left the viewing pose where we already are
                return True
            clamp = (step_i > 0 and remaining is not None and remaining > max_step)
            if clamp:
                move_pos = eef_pos + (target_pos - eef_pos) * (max_step / remaining)
                self.get_logger().info(
                    f"scan approach: marker {marker_id} viewing pose moved {remaining:.3f} m — "
                    f"stepping {max_step:.2f} m toward it."
                )
            else:
                # initial travel (unclamped, straight to the estimate's viewing
                # pose), or a correction already within one step
                move_pos = target_pos
            # take the final orientation so the marker's estimated spot is in frame
            goodPos, goodEuler = self.to_good_frame(move_pos, target_euler)
            if not self.move_to_pose(goodPos, goodEuler):
                return False
            if step_i > 0 and not clamp:
                # arrived at the corrected viewing pose
                return True
            # after the initial travel and each clamped step: brief window so a
            # sighting can correct the estimate before the next iteration. Not
            # seeing it here is fine — only scanToMarker's final observation is
            # required.
            self.allow_marker_updates(marker_id)
            try:
                time.sleep(1.0)
            finally:
                self.block_marker_updates()
        self.get_logger().error(
            f"scan approach: viewing pose for marker {marker_id} not reached within "
            f"15 correction steps of {max_step:.2f} m — estimate may be diverging. Aborting."
        )
        return False

    @_timed
    def moveToViewingDistance(self, marker_id, viewing_distance):
        """Go straight to marker_id's viewing pose at viewing_distance — one
        unclamped move, no update window, no detection requirement. Waypoint
        form: {'move': distance_m}. Use it to close distance quickly when the
        marker pose is already trusted; use {'scan': ...} when the pose needs
        to be (re)measured."""
        target_pos, target_euler = self._viewing_pose_for(marker_id, viewing_distance)
        if target_pos is None:
            self.get_logger().error(
                f"moveToViewingDistance: marker {marker_id} has no pose entry."
            )
            return False
        goodPos, goodEuler = self.to_good_frame(target_pos, target_euler)
        self.get_logger().info(
            f"moveToViewingDistance: marker {marker_id} — direct move to the "
            f"viewing pose at {viewing_distance:.3f} m."
        )
        return self.move_to_pose(goodPos, goodEuler)

    @_timed
    def scanToMarker(self, marker_id=0, viewing_distance=0.20):
        """Move the camera to face a known/estimated marker; extra passes
        re-aim at the refreshed pose so the measurement is head-on. The
        approach is step-limited (scan_approach_max_step) and re-observes the
        marker between steps, adapting to poor initial estimates."""
        move_ok = False
        marker_spotted = False
        for scan_pass in range(max(1, 1)):
            entry = self._find_marker_entry(marker_id)
            if entry is None:
                self.get_logger().error(f"Marker {marker_id} not found in found_markers. Register it first.")
                return False, False

            self.get_logger().info(
                f"Scanning marker {marker_id} (pass {scan_pass + 1}/{max(1, self.scan_passes)}): "
                f"approaching viewing pose at {viewing_distance:.3f} m"
            )
            move_ok = self._approach_viewing_pose(marker_id, viewing_distance)
            if not move_ok:
                break

            # raw-measurement logging is active only during this observation window
            self._scan_log_movement_id += 1
            self._scan_log_marker_id = marker_id
            self._scan_log_distance = viewing_distance
            # a fresh detection commits a NEW entry dict, so an identity change
            # against this snapshot means the marker was seen THIS window (a
            # stale pose from an earlier scan doesn't count)
            prev_entry_id = id(self.stream.found_markers.get(marker_id))
            # allow_marker_updates sleeps 0.5 s first so frames captured while
            # still moving drain before the window opens
            self.allow_marker_updates(marker_id)
            try:
                # poll instead of a blind sleep: the first camera-pose commit
                # takes 1-2+ s after arrival (TF + detect latency), and a fixed
                # sleep intermittently loses that race; polling exits as soon
                # as a real detection lands, so fast detections cost nothing
                observation_pause = 10.0 if self.collect_orientation_noise_data else 4.0
                deadline = time.time() + observation_pause
                while time.time() < deadline:
                    observed_entry = self.stream.found_markers.get(marker_id)
                    if (not self.collect_orientation_noise_data
                            and observed_entry is not None
                            and id(observed_entry) != prev_entry_id
                            and not observed_entry.get('estimated', False)):
                        break
                    time.sleep(0.25)
            finally:
                self.block_marker_updates()
                self._scan_log_marker_id = None
                self._scan_log_distance = None

            observed_entry = self.stream.found_markers.get(marker_id)
            marker_spotted = (observed_entry is not None
                              and id(observed_entry) != prev_entry_id
                              and not observed_entry.get('estimated', False))
            if not marker_spotted:
                # nothing fresh to re-aim at
                break

        observed_entry = self._find_marker_entry(marker_id)
        if not move_ok:
            print(f"[SCAN] Marker {marker_id}: movement FAILED (pose unreachable).")
        elif not marker_spotted:
            self.get_logger().error(
                f"[SCAN] Marker {marker_id}: NOT detected after moving to view position."
            )
            raise MarkerNotVisibleError(
                f"marker {marker_id} not detected at the scan pose "
                f"(viewing distance {viewing_distance:.3f} m)"
            )
        else:
            pos = observed_entry.get('positionInWorld', 'N/A')
            print(f"[SCAN] Marker {marker_id}: detected at {pos}")
        return move_ok, marker_spotted

    @_timed
    def scanLocationForMarkers(self, estimated_pos, estimated_orient=[0,0,0], viewing_distance=0.15, frame_name=None):
        """Move the camera to face an estimated marker location."""
        estimated_pos = np.array(estimated_pos)
        if frame_name is None:
            frame_name = f"{self.estimatedMarkerPrefix}0"

        offsetPos = np.array([0.0, 0.0, viewing_distance])
        offsetOri = np.array([0.0, 0.0, 0.0])

        markerBadPos, markerBadEuler = self.to_bad_frame(estimated_pos, estimated_orient)

        # place the camera frame (from TF) at the viewing distance; the EEF is
        # backed off by the fixed camera mount offset automatically
        badPos, badEuler = self._camera_view_pose(
            markerBadPos, markerBadEuler, offsetPos, offsetOri,
        )

        goodPos, goodEuler = self.to_good_frame(badPos, badEuler)
        self.get_logger().info(f'Scanning for markers at estimated position: {estimated_pos}')
        move_ok = self.move_to_pose(goodPos, goodEuler)
        if not move_ok:
            self.get_logger().error(
                f"scanLocationForMarkers: could not reach the viewing pose for {estimated_pos}."
            )
            return False
        # snapshot entry identities: every fresh detection commits a NEW dict
        # into found_markers, so a changed/new object during the window means a
        # marker was actually seen at this location
        before = {mid: id(e) for mid, e in self.stream.found_markers.items()}
        # unknown IDs may be here, so open the window for ANY marker
        self.allow_marker_updates()
        try:
            deadline = time.time() + 4.0
            while time.time() < deadline:
                for mid, e in list(self.stream.found_markers.items()):
                    if not e.get('estimated', False) and before.get(mid) != id(e):
                        self.get_logger().info(
                            f"scanLocationForMarkers: marker {mid} detected at location {estimated_pos}."
                        )
                        return True
                time.sleep(0.25)
        finally:
            self.block_marker_updates()
        self.get_logger().error(
            f"scanLocationForMarkers: NO marker detected at estimated position {estimated_pos}."
        )
        raise MarkerNotVisibleError(
            f"no marker detected at estimated position {list(np.round(estimated_pos, 3))}"
        )

    def scanMultipleLocations(self, locations, viewing_distance=0.15, pause_duration=2.0):
        """Scan multiple estimated marker locations sequentially. A location
        with no visible marker raises MarkerNotVisibleError (from
        scanLocationForMarkers); an unreachable viewing pose aborts (False)."""
        for i, location in enumerate(locations):
            if isinstance(location, tuple) and len(location) == 2:
                pos, orient = location
            else:
                pos = location
                orient = None

            frame_name = f"{self.estimatedMarkerPrefix}{i}"
            self.get_logger().info(f"Scanning location {i+1}/{len(locations)}: {pos}")

            success = self.scanLocationForMarkers(
                estimated_pos=pos,
                estimated_orient=orient,
                viewing_distance=viewing_distance,
                frame_name=frame_name
            )

            if not success:
                self.get_logger().error(
                    f"scanMultipleLocations: could not reach location {i+1}/{len(locations)}. Aborting."
                )
                return False
            time.sleep(pause_duration)
        return True

    # ---- Plate operations ----

    @_timed
    def moveToMarker(self, markerID=0):
        """Walk the marker's pickup waypoint list (scan/gripper entries included)."""
        waypoints = self._get_waypoints_for_marker(markerID, 'pickup')
        return self._follow_waypoints(markerID, waypoints, "moveToMarker")

    @_timed
    def pickupPlate(self, markerID=0):
        if not self.moveToMarker(markerID):
            self.get_logger().error(f"pickupPlate: moveToMarker failed for marker {markerID}.")
            return False
        # keep grasp + carry joints so placePlate can replay them
        grasp_joints = self._walk_grasp_joints
        lift_joints = self._current_arm_joints()
        if grasp_joints is not None and lift_joints is not None:
            self._pickup_replay = {'marker': markerID, 'grasp': grasp_joints,
                                   'lift': lift_joints}
        else:
            self._pickup_replay = None
        return True

    @_timed
    def placePlate(self, markerID=0):
        """Walk the marker's place list. If the pickup here was recorded, the
        moves before the release are replayed in joint space instead, pinning
        J6 so pose IK can't flip the plate into a collision (plain
        carry+descend only; longer descents are walked pose-based)."""
        place_waypoints = self._get_waypoints_for_marker(markerID, 'place')
        open_idx = next((i for i, wp in enumerate(place_waypoints)
                         if wp.get('gripper') == 'open'), None)
        pre_release = place_waypoints if open_idx is None else place_waypoints[:open_idx]
        replay = getattr(self, '_pickup_replay', None)
        custom_descent = sum(1 for wp in pre_release if 'pos' in wp) > 2
        if custom_descent and replay is not None:
            self.get_logger().warn(
                f"placePlate: custom placement descent configured for marker {markerID}; "
                "using pose-based placement instead of the wrist-continuous joint replay."
            )
            replay = None
        if replay is not None and replay.get('marker') == markerID:
            self.get_logger().info(
                f"placePlate: replaying recorded pickup joints for marker {markerID} (wrist-continuous)."
            )
            if not (self.move_to_configuration(replay['lift'])
                    and self.move_to_configuration(replay['grasp'])):
                # no pose-based fallback: IK could pick a flipped wrist and
                # drop the plate at an angle. Abort still holding it.
                self.get_logger().error(
                    "placePlate: wrist-continuous joint replay failed; aborting WITHOUT placing "
                    "to avoid an angled/flipped drop. Plate is still held — recover and re-run."
                )
                return False
            self.open_gripper()
            # Continue with the entries after the release (e.g. the withdraw).
            if open_idx is not None and open_idx + 1 < len(place_waypoints):
                return self._follow_waypoints(
                    markerID, place_waypoints[open_idx + 1:], "placePlate"
                )
            return True

        if not self._follow_waypoints(markerID, place_waypoints, "placePlate"):
            return False
        if open_idx is None:
            # Config without an explicit release entry — release at the end.
            self.open_gripper()
        return True

    @_timed
    def _scan_with_retries(self, marker_id, scan_distance, caller):
        """scanToMarker, retried at 0.85x and 0.70x distance if the move fails.
        A reached pose with no detection raises MarkerNotVisibleError (from
        scanToMarker), terminating the whole procedure."""
        for factor in (1.0, 0.85, 0.70):
            move_ok, _ = self.scanToMarker(
                marker_id=marker_id, viewing_distance=factor * scan_distance
            )
            if move_ok:
                return True
        self.get_logger().error(
            f"{caller}: could not reach marker {marker_id}. Aborting."
        )
        return False

    def transferPlate(self, source_id, dest_id, rescan_id):
        """Pick up from source, place at dest, pick up from rescan, place back
        at source. All motion (scans included) comes from the offset-config
        waypoint lists; aborts on the first failed step."""
        self.get_logger().info(
            f"transferPlate: source={source_id}, dest={dest_id}, rescan={rescan_id}"
        )

        # Step 1 – pick up from source
        self.get_logger().info(f"Step 1: picking up plate from marker {source_id}")
        _p = self._bambu_printers.get(source_id)
        if _p:
            _p.prepare_for_pickup()
        if not self.pickupPlate(markerID=source_id):
            self.get_logger().error(
                f"transferPlate: pickupPlate failed for marker {source_id}. Aborting."
            )
            return False

        # Step 2 – place at destination
        self.get_logger().info(f"Step 2: placing plate at marker {dest_id}")
        if not self.placePlate(markerID=dest_id):
            self.get_logger().error(
                f"transferPlate: placePlate failed for marker {dest_id}. Aborting."
            )
            return False
        _p = self._bambu_printers.get(dest_id)
        if _p:
            _p.home()

        # Step 3 – pick up from rescan printer
        self.get_logger().info(f"Step 3: picking up plate from marker {rescan_id}")
        _p = self._bambu_printers.get(rescan_id)
        if _p:
            _p.prepare_for_pickup()
        if not self.pickupPlate(markerID=rescan_id):
            self.get_logger().error(
                f"transferPlate: pickupPlate failed for marker {rescan_id}. Aborting."
            )
            return False

        # Step 4 – place back at source
        self.get_logger().info(f"Step 4: placing plate at marker {source_id}")
        if not self.placePlate(markerID=source_id):
            self.get_logger().error(
                f"transferPlate: placePlate failed for marker {source_id}. Aborting."
            )
            return False
        _p = self._bambu_printers.get(source_id)
        if _p:
            _p.home()

        self.get_logger().info("transferPlate: sequence complete.")
        return True

    @_timed
    def scanMarkerApproach(self, marker_id, viewing_distance=0.15):
        """Scan at progressively closer distances (1.75x down to 1.0x).
        A marker that isn't seen raises MarkerNotVisibleError; the first,
        longest distance gets one closer retry before that propagates."""
        distances = [
            (1.75 * viewing_distance, 2.0),
            (1.50 * viewing_distance, 1.0),
            (1.25 * viewing_distance, 1.0),
            (1.00 * viewing_distance, 1.0),
            (1.00 * viewing_distance, 1.0),
            (1.00 * viewing_distance, 1.0),
        ]

        for i, (dist, pause) in enumerate(distances):
            if i == 0:
                # the farthest pose is fragile on short-reach arms (lite6):
                # estimate noise can make it unreachable or off-frame. Retry
                # once closer before giving up on the whole approach.
                try:
                    _, spotted = self.scanToMarker(marker_id=marker_id, viewing_distance=dist)
                except MarkerNotVisibleError:
                    spotted = False
                if not spotted:
                    fallback = 1.4 * viewing_distance
                    self.get_logger().warn(
                        f"scanMarkerApproach: marker {marker_id} not seen at max distance "
                        f"({dist:.3f} m) — retrying closer at {fallback:.3f} m."
                    )
                    try:
                        self.scanToMarker(marker_id=marker_id, viewing_distance=fallback)
                    except MarkerNotVisibleError:
                        self.get_logger().error(
                            f"scanMarkerApproach: marker {marker_id} not seen at fallback distance "
                            f"({fallback:.3f} m) either — aborting approach."
                        )
                        raise
            else:
                self.scanToMarker(marker_id=marker_id, viewing_distance=dist)
            time.sleep(pause)

        return True

    def _scrape_dbg(self, msg):
        """Timestamped line to the scrape debug log. Instrumentation only."""
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self.get_logger().info(f"DBG {msg}")
        try:
            with open(os.path.join(_LOG_DIR, "scrape_debug.log"), "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _current_arm_joints(self):
        """Return the 6 arm joint angles (rad) in self.moveit2.joint_names order, or None."""
        js = self.moveit2.joint_state
        if js is None:
            return None
        names = list(js.name)
        try:
            return [float(js.position[names.index(j)]) for j in self.moveit2.joint_names]
        except ValueError:
            return None

    def _load_traj_samples(self, rel_path):
        """Parse a recorded joint trajectory (xArm-Studio .traj: '# frequency='
        header, then comma-separated joint radians at that fixed rate) into
        its demonstrated segment. Recording convention: travel to the
        segment's start pose, HOLD >=1 s, then demonstrate the path — only
        the part after the last >=1 s pause is used. Cached per path;
        (samples ndarray, frequency) or (None, None) on failure."""
        cached = self._traj_samples_cache.get(rel_path)
        if cached is not None:
            return cached
        path = rel_path if os.path.isabs(rel_path) else os.path.join(_DATA_DIR, rel_path)
        freq = 250.0
        rows = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('#'):
                        if 'frequency=' in line:
                            freq = float(line.split('frequency=')[1])
                        continue
                    vals = [float(v) for v in line.split(',') if v != '']
                    if len(vals) >= 6:
                        rows.append(vals[:6])
        except (OSError, ValueError) as ex:
            self.get_logger().error(f"_load_traj_samples: cannot read {path}: {ex}")
            return (None, None)
        if len(rows) < 2:
            self.get_logger().error(f"_load_traj_samples: no joint samples in {path}")
            return (None, None)
        joints = np.asarray(rows, dtype=float)

        # stillness = max joint speed under ~1.4 deg/s; the guide segment
        # starts after the last pause of at least 1 s
        still = np.abs(np.diff(joints, axis=0)).max(axis=1) < 1e-4
        min_run = int(freq)
        run_start, seg_start = None, 0
        for i, is_still in enumerate(still):
            if is_still and run_start is None:
                run_start = i
            elif not is_still and run_start is not None:
                if i - run_start >= min_run:
                    seg_start = i
                run_start = None
        segment = joints[seg_start:]
        self.get_logger().info(
            f"_load_traj_samples: {os.path.basename(path)} — {len(segment)} samples "
            f"({len(segment) / freq:.1f} s) from the segment after {seg_start / freq:.1f} s."
        )
        self._traj_samples_cache[rel_path] = (segment, freq)
        return (segment, freq)

    def _load_traj_guide(self, rel_path):
        """Sparse guide configs for the per-config fallback walk: the recorded
        segment downsampled so consecutive configs differ by at most
        traj_guide_step rad on any joint (final config always kept)."""
        cached = self._traj_guide_cache.get(rel_path)
        if cached is not None:
            return cached
        segment, _ = self._load_traj_samples(rel_path)
        if segment is None:
            return None
        configs = [segment[0]]
        for q in segment:
            if np.abs(q - configs[-1]).max() > self.traj_guide_step:
                configs.append(q)
        if np.abs(segment[-1] - configs[-1]).max() > 1e-3:
            configs.append(segment[-1])
        configs = [c.tolist() for c in configs]
        self._traj_guide_cache[rel_path] = configs
        return configs

    def _traj_guide_trajectory(self, rel_path):
        """The recorded segment as ONE JointTrajectory: demonstrated timing
        scaled by traj_guide_speed, points every traj_guide_point_dt of
        demonstrated time, velocities by finite differences (zero at the
        ends) so the controller splines through the points without stopping.
        The first point is pinned to the CURRENT joint state — move_group
        rejects trajectories that don't start at the robot's state, and the
        arm is only within move_to_configuration's loose tolerance of the
        recorded start. None on failure."""
        segment, freq = self._load_traj_samples(rel_path)
        if segment is None:
            return None
        current = self._current_arm_joints()
        if current is None:
            self.get_logger().warn("_traj_guide_trajectory: joint state unavailable.")
            return None

        stride = max(1, int(round(freq * self.traj_guide_point_dt)))
        idx = list(range(0, len(segment), stride))
        if idx[-1] != len(segment) - 1:
            idx.append(len(segment) - 1)
        if len(idx) < 2:
            return None
        pts = segment[idx]
        pts[0] = np.asarray(current[:pts.shape[1]], dtype=float)
        times = np.asarray(idx, dtype=float) / freq / max(self.traj_guide_speed, 1e-3)

        vels = np.zeros_like(pts)
        vels[1:-1] = (pts[2:] - pts[:-2]) / (times[2:] - times[:-2])[:, None]

        jt = JointTrajectory()
        jt.joint_names = list(self.moveit2.joint_names)[:pts.shape[1]]
        for q, v, t in zip(pts, vels, times):
            p = JointTrajectoryPoint()
            p.positions = q.tolist()
            p.velocities = v.tolist()
            p.time_from_start = Duration(seconds=float(t)).to_msg()
            jt.points.append(p)
        return jt

    def _replay_traj_guide(self, rel_path):
        """Replay a recorded joint path between procedure phases along its
        demonstrated (collision-free) route. Waypoint form: {'traj': path
        relative to data/}. The arm first travels to the recorded start via
        a planned move, then runs the whole recording as one continuous
        trajectory. Falls back to the per-config walk (jerky but planned) if
        the continuous execution can't be built or is rejected. False on
        failure — a half-followed guide leaves the arm off the known path,
        so the caller must abort rather than continue."""
        segment, _ = self._load_traj_samples(rel_path)
        if segment is None:
            return False
        if not self.move_to_configuration(segment[0].tolist()):
            self.get_logger().error(
                f"_replay_traj_guide: could not reach the recorded start of {rel_path}."
            )
            return False
        jt = self._traj_guide_trajectory(rel_path)
        if jt is not None:
            self.get_logger().info(
                f"_replay_traj_guide: executing {rel_path} as one continuous "
                f"trajectory ({len(jt.points)} points)."
            )
            if self.execute_joint_trajectory(jt):
                return True
            self.get_logger().warn(
                "_replay_traj_guide: continuous execution failed — falling back "
                "to the per-config walk."
            )
        configs = self._load_traj_guide(rel_path)
        if not configs:
            return False
        self.get_logger().info(
            f"_replay_traj_guide: walking {len(configs)} guide config(s) from {rel_path}."
        )
        for k, q in enumerate(configs):
            if not self.move_to_configuration(q):
                self.get_logger().error(
                    f"_replay_traj_guide: guide config {k+1}/{len(configs)} failed."
                )
                return False
        return True

    def _rotate_wrist(self, rotate_degrees):
        """Roll the wrist (last joint) by -rotate_degrees and back — dislodges
        debris / shifts the plate after a scrape. Waypoint form:
        {'rotate': degrees}. A failed rotation only warns and returns False;
        callers continue (placePlate's joint replay restores the wrist)."""
        self.get_logger().info(
            f"_rotate_wrist: rotating end-effector joint by {rotate_degrees:.1f}° and back."
        )
        current_joints = self._current_arm_joints()
        if current_joints is None:
            self.get_logger().warn("_rotate_wrist: joint state unavailable — skipping rotation.")
            return False
        rotated_joints = list(current_joints)
        rotated_joints[-1] -= np.radians(rotate_degrees)

        # J6 angle vs its limit (+/-180 mk3, +/-155 mk2)
        j6_now = np.degrees(current_joints[-1])
        j6_tgt = np.degrees(rotated_joints[-1])
        self._scrape_dbg(
            "ROTATE current_joints_deg=" + str(np.round(np.degrees(current_joints), 1).tolist())
        )
        self._scrape_dbg(
            f"ROTATE j6 {j6_now:.1f} deg --({-rotate_degrees:.0f})--> target {j6_tgt:.1f} deg; "
            f"exceeds_155={abs(j6_tgt) > 155.0} exceeds_180={abs(j6_tgt) > 180.0}"
        )

        rot_ok = self.move_to_configuration(rotated_joints)
        self._scrape_dbg(f"ROTATE move_to(rotated) ok={rot_ok}")
        time.sleep(0.5)
        # restore the wrist angle before any subsequent placing
        res_ok = self.move_to_configuration(current_joints)
        self._scrape_dbg(f"ROTATE move_to(restore) ok={res_ok} "
                         f"joints_after={np.round(np.degrees(self._current_arm_joints() or []),1).tolist()}")
        time.sleep(0.5)
        if not rot_ok:
            self.get_logger().warn(
                "_rotate_wrist: rotation failed. Continuing "
                "(placePlate replay will return the wrist to the grasp config)."
            )
        return rot_ok

    @_timed
    def scrapePlate(self, source_id, scrape_id, wait_after_pickup=False, wait_duration=60.0):
        """Pick up from source_id, scrape against the scrape_id surface, put
        it back. wait_after_pickup delays before scraping (cooldown). All
        motion — including any post-scrape wrist rotation ({'rotate': deg}
        entries) — comes from the marker's 'scrape' waypoint list."""
        scrape_waypoints = self._get_waypoints_for_marker(scrape_id, 'scrape')
        if not scrape_waypoints:
            self.get_logger().error(
                f"scrapePlate: no 'scrape' waypoints configured for marker {scrape_id}'s "
                "offset config. Aborting."
            )
            return False
        self.get_logger().info(
            f"scrapePlate: source={source_id}, scrape={scrape_id}, "
            f"{len(scrape_waypoints)} scrape waypoint(s)"
        )

        # Step 1 – pick up plate from source (its pickup list scans the marker first)
        self.get_logger().info(f"Step 1: picking up plate from marker {source_id}")
        if not self.pickupPlate(markerID=source_id):
            self.get_logger().error(
                f"scrapePlate: pickupPlate failed for marker {source_id}. Aborting."
            )
            return False
        if wait_after_pickup:
            self.get_logger().info(
                f"scrapePlate: waiting {wait_duration} s after pickup before scraping."
            )
            time.sleep(wait_duration)

        # markers are pinned by default — a close-range sighting can't corrupt
        # the scrape marker pose outside its own scan waypoint's window

        # log the marker pose the waypoints get applied to
        _e4 = self._find_marker_entry(scrape_id)
        if _e4 is not None:
            self._scrape_dbg(
                f"SCRAPE marker {scrape_id} pos={np.round(_e4['positionInBase'],4)} "
                f"euler_deg={np.round(np.degrees(_e4['eulerInBase']),2)} estimated={_e4.get('estimated')}"
            )

        # Step 2 – scrape (a failed final retract only warns so the plate
        # can still be returned)
        self.get_logger().info(f"Step 2: walking {len(scrape_waypoints)} scrape waypoint(s)")
        if not self._follow_waypoints(scrape_id, scrape_waypoints, "scrapePlate",
                                      tolerant_last=True):
            self.get_logger().error(
                f"scrapePlate: scrape waypoints failed for marker {scrape_id}. Aborting."
            )
            return False
        self._scrape_dbg(
            "SCRAPE walk done joints_deg=" +
            str(np.round(np.degrees(self._current_arm_joints() or []), 1).tolist())
        )

        # Step 3 – place back at source; its place list's scan waypoint opens
        # the update window that refreshes the source marker pose
        self.get_logger().info(f"Step 3: placing plate back at marker {source_id}")
        if not self.placePlate(markerID=source_id):
            self.get_logger().error(
                f"scrapePlate: placePlate failed for marker {source_id}. Aborting."
            )
            return False

        self.get_logger().info("scrapePlate: sequence complete.")
        return True

    def pickupOnly(self, source_id, wait_after_pickup=False, wait_duration=60.0):
        """Approach marker source_id and pick up the plate, then stop. Same
        pickup step as scrapePlate's Step 1 — the plate is left in the gripper
        (no scrape, no place-back)."""
        self.get_logger().info(f"pickupOnly: source={source_id}")

        # Step 1 – pick up plate from source (its pickup list scans the marker first)
        self.get_logger().info(f"Step 1: picking up plate from marker {source_id}")
        if not self.pickupPlate(markerID=source_id):
            self.get_logger().error(
                f"pickupOnly: pickupPlate failed for marker {source_id}. Aborting."
            )
            return False
        if wait_after_pickup:
            self.get_logger().info(
                f"pickupOnly: waiting {wait_duration} s after pickup."
            )
            time.sleep(wait_duration)

        self.get_logger().info("pickupOnly: sequence complete.")
        return True

    def _collect_marker_in_base(self, marker_id, window=1.5):
        """Average the raw (unfiltered) marker detections over `window` seconds.
        Returns (mean_pos_in_base, mean_pos_in_camera, n_samples); n=0 if the
        marker wasn't seen. Opens its own update window so detection runs."""
        self._calib_buf = []
        self._calib_collect_id = marker_id
        self.allow_marker_updates(marker_id)
        try:
            deadline = time.time() + window
            while time.time() < deadline:
                time.sleep(0.05)
        finally:
            self.block_marker_updates()
            self._calib_collect_id = None
        buf = self._calib_buf
        self._calib_buf = []
        if not buf:
            return None, None, 0
        mean_base = np.mean(np.stack([b for b, _c in buf]), axis=0)
        mean_cam = np.mean(np.stack([c for _b, c in buf]), axis=0)
        return mean_base, mean_cam, len(buf)

    def _open_calibration_csv(self, marker_id):
        """Open a timestamped per-pose CSV under data/logs and write the column
        header. Rows are appended + flushed as each pose is captured, so an
        interrupted run still keeps every pose collected so far. Returns
        (file, writer, path) or (None, None, None) on failure."""
        try:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(_LOG_DIR, f"camera_offset_calib_m{marker_id}_{stamp}.csv")
            f = open(path, "w", newline="")
            w = csv.writer(f)
            w.writerow(["pose_i", "tilt_deg", "az_deg", "dist_m", "n_frames",
                        "Pmeas_x", "Pmeas_y", "Pmeas_z",
                        "Pcam_x", "Pcam_y", "Pcam_z",
                        "eef_x", "eef_y", "eef_z",
                        "q_be_x", "q_be_y", "q_be_z", "q_be_w"])
            f.flush()
            self.get_logger().info(f"calibrate: streaming per-pose data to {path}")
            return f, w, path
        except Exception as ex:
            self.get_logger().warn(f"_open_calibration_csv: could not open CSV ({ex}).")
            return None, None, None

    def _write_calibration_row(self, writer, f, s):
        """Append one captured pose to the CSV and flush immediately."""
        if writer is None:
            return
        P, C, e, q = s['P'], s['P_cam'], s['eef_pos'], s['quat']
        writer.writerow([s['i'], s['tilt'], s['az'], round(s['dist'], 4), s['n'],
                         round(float(P[0]), 6), round(float(P[1]), 6), round(float(P[2]), 6),
                         round(float(C[0]), 6), round(float(C[1]), 6), round(float(C[2]), 6),
                         round(float(e[0]), 6), round(float(e[1]), 6), round(float(e[2]), 6),
                         round(float(q[0]), 6), round(float(q[1]), 6),
                         round(float(q[2]), 6), round(float(q[3]), 6)])
        f.flush()

    def _finalize_calibration_csv(self, writer, f, per_pose_resid, delta, P_true,
                                  cond, before_rms, after_rms, t_cur,
                                  new_total_correction, eps=None, scale=None):
        """Append the fit summary (and per-pose residuals) after the solve."""
        if writer is None:
            return
        try:
            writer.writerow([])
            writer.writerow(["# per_pose_fit_resid_m", *np.round(per_pose_resid, 6).tolist()])
            writer.writerow(["# delta_eef_m", *np.round(delta, 6).tolist()])
            if eps is not None:
                writer.writerow(["# eps_cam_deg", *np.round(np.degrees(eps), 4).tolist()])
            if scale is not None:
                writer.writerow(["# depth_scale_err", round(float(scale), 6)])
            writer.writerow(["# P_true_base_m", *np.round(P_true, 6).tolist()])
            writer.writerow(["# current_net_t_ec_m",
                             *(np.round(t_cur, 6).tolist() if t_cur is not None else [])])
            writer.writerow(["# new_total_correction_m", *np.round(new_total_correction, 6).tolist()])
            writer.writerow(["# condition_number", round(float(cond), 3)])
            writer.writerow(["# spread_rms_before_m", round(before_rms, 6),
                             "spread_rms_after_m", round(after_rms, 6)])
            f.flush()
        except Exception as ex:
            self.get_logger().warn(f"_finalize_calibration_csv: could not write summary ({ex}).")

    def calibrate_camera_offset(self, marker_id, viewing_distances=(0.25, 0.20, 0.15),
                                tilt_degs=(12.0, 18.0), azimuths_deg=(0.0, 90.0, 180.0, 270.0),
                                apply=False):
        """Estimate the eef->camera translation error from one STATIONARY marker.

        Orbits the camera on a cone around the marker (keeping it centred while
        varying wrist orientation) and records, per pose, the measured
        marker-in-base pose P_i and the base<-eef rotation R_i. For a stationary
        marker the true position P is constant, so P = P_i + R_i @ delta. Least-
        squares over the orientation-diverse poses solves the 6 unknowns
        [P, delta]; delta is the correction (EEF frame) to add to the camera
        mount translation. Reports delta and the corrected net translation.

        viewing_distances is swept FAR->NEAR so the marker is acquired well
        inside the FOV at the first (farthest) pose before the closer, more
        tilted poses run (like the scan/approach paths). Pass them large-to-small.

        apply=False (default) only reports. apply=True installs delta for the
        rest of the session (both scan targeting and measured marker poses).
        Returns delta (3,) or None on failure.
        """
        entry = self._find_marker_entry(marker_id)
        if entry is None:
            self.get_logger().error(
                f"calibrate_camera_offset: marker {marker_id} not found. Scan it first.")
            return None
        marker_pos = np.asarray(entry['positionInBase'], dtype=float)
        marker_euler = np.asarray(entry['eulerInBase'], dtype=float)
        offsetOri = np.asarray(self.offsetOri, dtype=float)

        # measure raw (uncorrected) during the sweep; restore/replace afterwards
        prior_correction = self._camera_offset_correction
        self._camera_offset_correction = None

        # build the cone of viewing poses, distance-major FAR->NEAR: acquire the
        # marker head-on at the farthest standoff (safely in-FOV) first, then
        # work inward, adding the tilted views (orientation diversity) at each
        # distance. Closer+tilted poses that clip the FOV just get skipped, but
        # the farther samples at that azimuth still cover them.
        dists = sorted({float(d) for d in viewing_distances}, reverse=True)
        poses = []
        for dist in dists:
            poses.append((0.0, 0.0, dist))          # head-on at this standoff
            for th in tilt_degs:
                for az in azimuths_deg:
                    poses.append((th, az, dist))

        # open the CSV up front so each pose is persisted as it lands
        csv_file, csv_writer, save_path = self._open_calibration_csv(marker_id)

        samples = []  # dicts: pose params, P_meas, base<-eef rotation, eef FK pos
        try:
            for i, (theta_deg, azim_deg, dist) in enumerate(poses):
                # tilt the view direction theta towards azimuth az with ZERO net
                # roll about the optical axis (conjugated rotation). A plain
                # Rz(az)*Ry(tilt) would also spin the wrist by the full azimuth
                # (90/180/270 deg!) between poses.
                Rcone = (R.from_euler('z', azim_deg, degrees=True)
                         * R.from_euler('y', theta_deg, degrees=True)
                         * R.from_euler('z', -azim_deg, degrees=True))
                offset_pos = Rcone.apply([0.0, 0.0, dist])
                offset_ori = (Rcone * R.from_euler("XYZ", offsetOri, degrees=False)).as_euler("XYZ")

                eef_bad_pos, eef_bad_euler = self._camera_view_pose(
                    marker_pos, marker_euler, offset_pos, offset_ori)
                goodPos, goodEuler = self.to_good_frame(eef_bad_pos, eef_bad_euler)

                self.get_logger().info(
                    f"calibrate: pose {i+1}/{len(poses)} (tilt={theta_deg:.0f} deg "
                    f"az={azim_deg:.0f} deg dist={dist:.3f})")
                move_ok = self.move_to_pose(goodPos, goodEuler)
                if not move_ok:
                    self.get_logger().warn("calibrate: pose unreachable, skipping.")
                    continue

                time.sleep(0.5)  # let the low-pass settle on the fresh view
                P_meas, P_cam, n = self._collect_marker_in_base(marker_id, window=1.5)
                eef_pos, R_be = self._base_from_eef_transform()
                if P_meas is None or R_be is None:
                    self.get_logger().warn(
                        f"calibrate: marker not seen / no TF at this pose "
                        f"(samples={n}), skipping.")
                    continue
                s = {
                    'i': i, 'tilt': theta_deg, 'az': azim_deg, 'dist': dist,
                    'n': n, 'P': np.asarray(P_meas, dtype=float),
                    'P_cam': np.asarray(P_cam, dtype=float),
                    'R': R_be.as_matrix(), 'quat': R_be.as_quat(),
                    'eef_pos': eef_pos,
                }
                samples.append(s)
                self._write_calibration_row(csv_writer, csv_file, s)  # flush per pose
                # ideal view: marker centred at [0,0,dist] in the camera frame;
                # the deviation shows which camera axis is off at this pose
                cam_dev = np.asarray(P_cam, dtype=float) - np.array([0.0, 0.0, dist])
                self.get_logger().info(
                    f"calibrate:   captured P_base={np.round(P_meas,4).tolist()} "
                    f"P_cam={np.round(P_cam,4).tolist()} "
                    f"(cam deviation from ideal {np.round(cam_dev,4).tolist()}, "
                    f"{n} raw frames) -> saved")
        finally:
            # sweep done; restore prior state (apply step below may overwrite)
            self._camera_offset_correction = prior_correction

        if len(samples) < 3:
            if csv_file is not None:
                csv_file.close()
            self.get_logger().error(
                f"calibrate_camera_offset: only {len(samples)} usable pose(s); "
                "need >=3 with varied orientation. Aborting. "
                f"(raw poses saved to {save_path})")
            return None

        # Extended linear model. True camera extrinsic errors: translation
        # delta (EEF frame), small-angle rotation eps (camera frame), and a
        # depth/size scale s (aruco range scales with the configured marker
        # size, so a size error appears as a range-proportional bias). With
        # p_i the marker measured in the camera frame:
        #   P                    = t_i + R_i (t_ec + delta + R_ec (I+[eps]x)(1+s) p_i)
        #   P_meas_i (pipeline)  = t_i + R_i (t_ec + R_ec p_i)
        #   => P_meas_i = P - R_i delta + R_i R_ec [p_i]x eps - (R_i R_ec p_i) s
        # Linear in x = [P(3), delta(3), eps(3), s] -> one lstsq. Separating
        # eps and s stops them from contaminating delta (they otherwise show
        # up as a spurious range-dependent translation).
        R_ec_rot = self._eef_to_camera_rotation()
        R_ec = R_ec_rot.as_matrix() if R_ec_rot is not None else None
        n_unk = 10 if R_ec is not None else 6
        if R_ec is None:
            self.get_logger().warn(
                "calibrate: eef->camera rotation unavailable from TF — falling "
                "back to translation-only model (rotation/scale errors will "
                "leak into delta).")
        A = np.zeros((3 * len(samples), n_unk))
        b = np.zeros(3 * len(samples))
        for k, s in enumerate(samples):
            A[3*k:3*k+3, 0:3] = np.eye(3)
            A[3*k:3*k+3, 3:6] = -s['R']
            if R_ec is not None:
                p = s['P_cam']
                px = np.array([[0.0, -p[2], p[1]],
                               [p[2], 0.0, -p[0]],
                               [-p[1], p[0], 0.0]])
                RiRec = s['R'] @ R_ec
                A[3*k:3*k+3, 6:9] = RiRec @ px
                A[3*k:3*k+3, 9] = -(RiRec @ p)
            b[3*k:3*k+3] = s['P']
        # rcond=None truncates near-null directions (e.g. roll about the
        # optical axis, unobservable with the marker centred) to the
        # minimum-norm solution instead of letting them blow up.
        x, _res, rank, sing = np.linalg.lstsq(A, b, rcond=None)
        P_true = x[0:3]
        delta = x[3:6]
        eps = x[6:9] if n_unk == 10 else None      # rad, camera frame
        scale = float(x[9]) if n_unk == 10 else None  # fractional depth error
        # condition of the retained (rank) subspace; sing[rank-1] is the
        # smallest singular value actually used
        cond = sing[0] / sing[rank - 1] if rank > 0 and sing[rank - 1] > 0 else np.inf

        # error signatures: raw spread (before) vs fit residual (after)
        meas = np.stack([s['P'] for s in samples])
        before_rms = float(np.sqrt(np.mean(np.sum((meas - meas.mean(axis=0))**2, axis=1))))
        pred = (A @ x).reshape(-1, 3)
        per_pose_resid = np.linalg.norm(meas - pred, axis=1)  # m, one per sample
        after_rms = float(np.sqrt(np.mean(per_pose_resid**2)))

        t_cur = self._eef_to_camera_translation()
        base = prior_correction if prior_correction is not None else np.zeros(3)
        new_total_correction = base + delta
        corrected_net = (t_cur + new_total_correction) if t_cur is not None else None

        # append the fit summary to the already-streamed CSV, then close it
        self._finalize_calibration_csv(
            csv_writer, csv_file, per_pose_resid, delta, P_true, cond,
            before_rms, after_rms, t_cur, new_total_correction,
            eps=eps, scale=scale)
        if csv_file is not None:
            csv_file.close()

        eps_txt = (f"{np.round(np.degrees(eps), 2).tolist()} deg (camera frame)"
                   if eps is not None else "not fitted (no R_ec)")
        scale_txt = (f"{scale*100:+.2f} % (negative => camera reports depth LONG => "
                     "configured marker size too large; actual size ~= configured * (1+s))"
                     if scale is not None else "not fitted (no R_ec)")
        self.get_logger().info(
            "calibrate_camera_offset RESULT:\n"
            f"  usable poses      : {len(samples)}/{len(poses)}\n"
            f"  delta (EEF, m)    : {np.round(delta, 4).tolist()}  (this run)\n"
            f"  |delta|           : {np.linalg.norm(delta)*1000:.1f} mm\n"
            f"  mount rot eps     : {eps_txt}\n"
            f"  depth scale err   : {scale_txt}\n"
            f"  marker spread RMS : before={before_rms*1000:.1f} mm  after={after_rms*1000:.1f} mm\n"
            f"  A condition #     : {cond:.1f} (retained subspace, rank {rank}/{n_unk}; "
            f"large => poor pose diversity)\n"
            f"  current net t_ec  : {np.round(t_cur,4).tolist() if t_cur is not None else 'N/A'} m\n"
            f"  corrected net t_ec: {np.round(corrected_net,4).tolist() if corrected_net is not None else 'N/A'} m\n"
            f"  data saved to     : {save_path}\n"
            f"  -> to make permanent: fix the marker size / intrinsics first if the "
            f"scale error is large, then shift the camera mount origin in the URDF "
            f"by delta (EEF frame) and rotate it by eps (camera frame)."
        )
        if after_rms > before_rms:
            self.get_logger().warn(
                "calibrate_camera_offset: fit did NOT reduce the marker spread — "
                "the residual may be orientation error, not translation. Treat delta with caution.")

        if apply:
            self._camera_offset_correction = new_total_correction
            self.get_logger().warn(
                f"calibrate_camera_offset: APPLIED correction "
                f"{np.round(new_total_correction,4).tolist()} m for this session "
                "(scan targeting + measured marker poses).")
        else:
            self.get_logger().info(
                "calibrate_camera_offset: report only (apply=False); nothing changed.")
        return delta

    @_timed
    def go_home(self, velocity_scaling=0.2):
        """Move all joints to zero. Plans from the actual encoder state, so it
        corrects drift from lost steps; kept slow to avoid further stalls."""
        self.get_logger().warn(
            f"go_home: resyncing to home position (velocity_scaling={velocity_scaling}). "
            "Planning from actual encoder state to correct any step-loss drift."
        )
        # settle before reading the pose
        time.sleep(0.5)

        home_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        prev_velocity = self.moveit2.max_velocity
        prev_acceleration = self.moveit2.max_acceleration
        try:
            self.moveit2.max_velocity = velocity_scaling
            self.moveit2.max_acceleration = velocity_scaling
            self.move_to_configuration(home_joints)
            time.sleep(self.move_settle_delay)
        finally:
            self.moveit2.max_velocity = prev_velocity
            self.moveit2.max_acceleration = prev_acceleration

        self.get_logger().info("go_home: reached home position.")


