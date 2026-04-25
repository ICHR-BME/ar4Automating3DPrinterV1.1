from ArucoDetector import ArucoDetectionViewer
import rclpy
import numpy as np
import os
import json
import csv
import datetime
import functools
from pymoveit2 import GripperInterface
from printerclass import BambuPrinter
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
import tf2_ros
from simulated3DPrinter import Simulated3DPrinter
import time
import threading


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _timed(method):
    """Decorator that records wall-clock duration and full call chain for each public method."""
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
                 color_topic="/rgbd_camera/image", depth_topic="/rgbd_camera/depth_image", camera_info_topic="/rgbd_camera/camera_info",
                 feed_rotation_deg=0.0, marker_sizes=None):
        self._startup_start = time.perf_counter()
        super().__init__(source=stream_source,
                         camera_index=camera_index,
                         camera_keyword=camera_keyword,
                         color_topic=color_topic,
                         depth_topic=depth_topic,
                         camera_info_topic=camera_info_topic,
                         feed_rotation_deg=feed_rotation_deg,
                         marker_sizes=marker_sizes,
                         calibration_file=os.path.join(_SCRIPT_DIR, "camera_matrix.npz"))
        self.get_logger().info(f"printerAutomation initialized, calibration_mode={calibration_mode}")

        # Estimated marker frame name prefix
        self.estimatedMarkerPrefix = "estimated_marker_"

        # When True, open_gripper/close_gripper are no-ops (workaround for sim instability)
        self.gripper_disabled = False
        # When True, register_estimated_marker adds random noise to test scan robustness
        self.randomize_estimated_markers = False
        # When True, scanToMarker pauses for 10 s after arriving to collect raw orientation
        # noise data instead of the normal 1 s observation window.
        self.collect_orientation_noise_data = False

        ## For the small handle
        #self.markerToHandleOffset = np.array([0.0, 0.05, 0.06])
        #self.markerToPickupOffset = np.array([0.0, 0.20, 0.06])
        
        ## For the big handle
        #self.markerToHandleOffset = np.array([0.0, 0.033, 0.1])
        #self.markerToPickupOffset = np.array([0.0, 0.125, 0.1])

        ## For the 3D printer with the marker to the side]
        #self.markerToHandleOffset = np.array([0.09, 0.05, 0.22])
        #self.markerToPickupOffset = np.array([0.09, 0.155, 0.22])


        ## Named offset configs: each entry maps a name to handle and pickup offsets
        ## in the marker's local frame. Add new entries here for different printer types.
        self.offset_configs = {
            # Printer with the handle above the marker
            'printer_offset': {
                'handleOffset': np.array([0.0, 0.067, 0.077]),
                'pickupOffset': np.array([0.0, 0.167, 0.077]),
            },
            # Printer with the marker to the side
            'box_offset': {
                'handleOffset': np.array([0.0, 0.05, 0.102]),
                'pickupOffset': np.array([0.0, 0.15, 0.102]),
            },
        }
        ## Map marker_id -> config name. IDs not listed fall back to 'box_offset'.
        self.marker_offset_config = {}

        ## Offset from the scrape marker origin in the marker's local frame [x, y, z]
        ## used by scrapePlate().  z is the closest approach distance along the marker's
        ## outward Z axis; x/y shift the contact point laterally in the marker plane.
        self.scrape_offset = np.array([0.0, 0.12, 0.13])

        self.offsetOri = np.array([0.0, np.pi, np.pi / 2])

        # The camera is mounted below the gripper. Raise the end effector by this
        # amount (metres, base-link Z) when scanning so the camera aligns with the marker.
        self.camera_z_offset = 0.06

        # Persistent state save file — written every 5 s and loaded at startup
        self._state_save_path = os.path.join(_SCRIPT_DIR, "printer_state.json")
        self.create_timer(5.0, self._auto_save_state)

        # Timing log: one row per public-method call
        _timing_dir = os.path.join(_SCRIPT_DIR, "timingData")
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

        # Raw-measurement scan log (one row per video frame, written immediately).
        # Opened in write mode so old data is discarded on every restart.
        self._scan_log_path = os.path.join(_SCRIPT_DIR, "scan_raw_measurements.csv")
        self._scan_log_marker_id = None   # set by scanToMarker while active
        self._scan_log_distance = None
        self._scan_log_movement_id = 0     # increments each time a new scanToMarker observation window starts
        self._scan_log_file = open(self._scan_log_path, 'w', newline='')
        self._scan_log_writer = csv.writer(self._scan_log_file)
        self._scan_log_writer.writerow([
            'marker_id', 'scan_distance', 'movement_id',
            'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw',
            'cam_px', 'cam_py', 'cam_pz', 'cam_qx', 'cam_qy', 'cam_qz', 'cam_qw',
        ])
        self._scan_log_file.flush()

        # BambuPrinter integration: maps marker_id -> BambuPrinter instance.
        # Populate via register_bambu_printer() after constructing the node.
        self._bambu_printers: dict = {}

        # Gripper interface
        self.gripper = GripperInterface(
            node=self,
            gripper_joint_names=["gripper_jaw1_joint"],
            open_gripper_joint_positions=[0.00],
            closed_gripper_joint_positions=[0.0145],
            gripper_group_name="ar_gripper",
            callback_group=self._cb_group,
            gripper_command_action_name="gripper_controller/gripper_cmd",
        )

    def _record_timing(self, call_chain: str, duration_s: float):
        """Append one timing row (with full call chain) to the session CSV."""
        self._timing_writer.writerow(
            [datetime.datetime.now().isoformat(), call_chain, duration_s]
        )
        self._timing_file.flush()

    def record_startup_time(self):
        """Record the elapsed time from __init__ to now as a 'startup' row in the timing CSV."""
        elapsed = round(time.perf_counter() - self._startup_start, 4)
        self._record_timing("startup", elapsed)

    def pause_timing(self):
        """Pause the timing clock. Time elapsed while paused is excluded from all active timers."""
        if self._timing_pause_start is None:
            self._timing_pause_start = time.perf_counter()

    def resume_timing(self):
        """Resume the timing clock after a pause_timing() call."""
        if self._timing_pause_start is not None:
            self._timing_total_paused += time.perf_counter() - self._timing_pause_start
            self._timing_pause_start = None

    # ---- State persistence ----

    def save_state(self):
        """Serialise detected marker poses, offset config, and printer configs to JSON."""
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
                "marker_size": float(entry.get('marker_size', 0.03)),
                "estimated": bool(entry.get('estimated', False)),
            })
        try:
            with open(self._state_save_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"save_state: could not write {self._state_save_path}: {e}")

    def register_printers(self, printers):
        """Store printer configs so they are included in every save_state() call.

        *printers* is a list of dicts, each with keys:
          marker_id, pos, orient, door_marker_texture
        """
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
        """Restore marker poses and offset config from a previous save file.

        Each saved marker is registered as an estimated pose so the camera
        will overwrite it with a real detection on the next scan.  Returns
        True if a file was found and loaded, False otherwise.
        """
        if not os.path.exists(self._state_save_path):
            self.get_logger().info(f"load_state: no save file at {self._state_save_path} — starting fresh")
            return False
        try:
            with open(self._state_save_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.get_logger().warn(f"load_state: could not read {self._state_save_path}: {e}")
            return False

        # Restore offset config (JSON keys are always strings)
        for k, v in data.get("marker_offset_config", {}).items():
            self.marker_offset_config[int(k)] = v

        # Restore marker poses, preserving whether each was a real detection or an estimate
        for m in data.get("markers", []):
            marker_id = int(m["id"])
            pos = np.array(m["positionInBase"], dtype=float)
            euler = np.array(m["eulerInBase"], dtype=float)
            self.register_estimated_marker(marker_id=marker_id, bad_pos=pos, bad_euler=euler)
            # register_estimated_marker always marks entries as estimated=True.
            # If the saved entry was a real camera detection, restore that flag so
            # downstream code (e.g. runDoubleTransfer) won't overwrite it with a
            # geometric fallback.
            if not m.get("estimated", True):
                self.stream.found_markers[marker_id]['estimated'] = False

        # Restore printer configs so future save_state() calls preserve them
        if data.get("printers"):
            self._saved_printer_configs = data["printers"]

        n = len(data.get("markers", []))
        self.get_logger().info(
            f"load_state: restored {n} marker(s) and offset config from {self._state_save_path}"
        )
        return True

    def _auto_save_state(self):
        """Timer callback: persist state to disk every 5 s."""
        self.save_state()

    # ---- Raw measurement logging ----

    def _on_raw_marker_measurement(self, marker_id, pos_in_base, quat_in_base,
                                    pos_in_camera, quat_in_camera):
        """Called by ArucoDetector for every video frame that detects a marker.
        Writes one CSV row immediately (no buffering) so data is saved as it arrives.
        Only logs while a scanToMarker call is active (_scan_log_marker_id is set).
        """
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
        t.header.frame_id = "base_link"
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
        """
        Compute a target pose by applying an offset in the marker's local frame.

        Returns (target_pos, target_euler) in base_link.
        """
        R_marker = R.from_euler("XYZ", marker_euler, degrees=False)
        target_pos = marker_pos + R_marker.apply(offset_pos)
        target_euler = (R_marker * R.from_euler("XYZ", offset_ori, degrees=False)).as_euler("XYZ", degrees=False)
        return target_pos, target_euler

    def _move_to_marker_offset(self, marker_id, offset_pos, offset_ori=None):
        """
        Core routine: find marker, compute offset, and move.

        Returns True on success, False if marker not found.
        """
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
        self.freeze_markers()
        move_ok = self.move_to_pose(goodPos, goodEuler)
        self.unfreeze_markers()
        return move_ok

    def _get_offsets_for_marker(self, marker_id):
        """Return (handleOffset, pickupOffset) for the given marker ID."""
        config_name = self.marker_offset_config.get(marker_id, 'box_offset')
        config = self.offset_configs[config_name]
        return config['handleOffset'], config['pickupOffset']

    # ---- BambuPrinter integration ----

    def register_bambu_printer(self, marker_id, printer: BambuPrinter):
        """Associate an already-connected BambuPrinter instance with a marker ID.

        When enabled, transferPlate will command this physical printer to move
        its tool head to max X/Z before the robot picks up the build plate,
        and home it after the plate has been placed and the robot has withdrawn.
        """
        self._bambu_printers[marker_id] = printer
        self.get_logger().info(
            f"register_bambu_printer: marker {marker_id} → printer {printer.serial} at {printer.ip}"
        )
        printer.homing()

    # ---- Gripper ----

    def open_gripper(self):
        if self.gripper_disabled:
            self.get_logger().info("Gripper disabled — skipping open.")
            return
        self.get_logger().info("Opening gripper...")
        self.gripper.open()

    def close_gripper(self):
        if self.gripper_disabled:
            self.get_logger().info("Gripper disabled — skipping close.")
            return
        self.get_logger().info("Closing gripper...")
        self.gripper.close()

    # ---- Marker updates ----

    def freeze_markers(self):
        """Disable marker pose updates. Call before moving the robot."""
        self.stream.marker_updates_enabled = False
        self.get_logger().info("Marker pose updates frozen.")

    def unfreeze_markers(self):
        """Re-enable marker pose updates. Call after the robot has stopped."""
        # Delay to discard frames that were captured during movement (camera has
        # a slight pipeline delay, so in-flight frames arrive after the move ends).
        time.sleep(0.5)
        self.stream.marker_updates_enabled = True
        self.get_logger().info("Marker pose updates resumed.")

    # ---- Marker registration & scanning ----

    def register_estimated_marker(self, marker_id, bad_pos, bad_euler):
        """
        Pre-populate an estimated marker pose in both the TF tree and found_markers.

        Once the camera actually detects this marker, _enrich_marker_pose will
        overwrite both the TF frame and the found_markers entry with the real
        measurement.
        """
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

    @_timed
    def scanToMarker(self, marker_id=0, viewing_distance=0.20):
        """Move the camera to face a known/estimated marker using its TF frame."""
        entry = self._find_marker_entry(marker_id)
        if entry is None:
            self.get_logger().error(f"Marker {marker_id} not found in found_markers. Register it first.")
            return False

        offsetPos = np.array([0.0, 0.0, viewing_distance])
        badPos, badEuler = self._apply_offset_in_marker_frame(
            entry['positionInBase'], entry['eulerInBase'], offsetPos, self.offsetOri,
        )
        # Shift up in base-link Z so the camera (below the gripper) faces the marker
        badPos = badPos + np.array([0.0, 0.0, self.camera_z_offset])

        goodPos, goodEuler = self.to_good_frame(badPos, badEuler)
        self.get_logger().info(f"Scanning marker {marker_id}: moving to viewing pos={goodPos}")
        self.freeze_markers()
        move_ok = self.move_to_pose(goodPos, goodEuler)
        self.unfreeze_markers()

        # Activate raw-measurement logging for this marker/distance while the camera
        # observes.  Logging is stopped as soon as the observation window ends.
        self._scan_log_movement_id += 1
        self._scan_log_marker_id = marker_id
        self._scan_log_distance = viewing_distance
        # Pause to allow camera to observe the marker
        observation_pause = 10.0 if self.collect_orientation_noise_data else 1.0
        time.sleep(observation_pause)
        self._scan_log_marker_id = None
        self._scan_log_distance = None

        observed_entry = self._find_marker_entry(marker_id)
        marker_spotted = observed_entry is not None and not observed_entry.get('estimated', False)
        if not move_ok:
            print(f"[SCAN] Marker {marker_id}: movement FAILED (pose unreachable).")
        elif not marker_spotted:
            print(f"[SCAN] Marker {marker_id}: NOT detected after moving to view position.")
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

        badPos, badEuler = self._apply_offset_in_marker_frame(
            markerBadPos, markerBadEuler, offsetPos, offsetOri,
        )
        # Shift up in base-link Z so the camera (below the gripper) faces the marker
        badPos = badPos + np.array([0.0, 0.0, self.camera_z_offset])

        goodPos, goodEuler = self.to_good_frame(badPos, badEuler)
        self.get_logger().info(f'Scanning for markers at estimated position: {estimated_pos}')
        self.freeze_markers()
        self.move_to_pose(goodPos, goodEuler)
        self.unfreeze_markers()
        return True

    def scanMultipleLocations(self, locations, viewing_distance=0.15, pause_duration=2.0):
        """Scan multiple estimated marker locations sequentially."""
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

            if success:
                time.sleep(pause_duration)
                markers = self.marker_poses
                if markers:
                    self.get_logger().info(f"Detected {len(markers)} markers at location {i+1}")
                else:
                    self.get_logger().info(f"No markers detected at location {i+1}")

    # ---- Plate operations ----

    def _move_to_approach(self, markerID, approach_standoff=0.15):
        """Move to the standoff position along the marker Z axis (no gripper action)."""
        handle_offset, _ = self._get_offsets_for_marker(markerID)
        approach_offset = np.array([handle_offset[0], handle_offset[1], approach_standoff])
        return self._move_to_marker_offset(markerID, approach_offset)

    @_timed
    def moveToMarker(self, markerID=0, approach_standoff=0.15):
        self.open_gripper()
        handle_offset, _ = self._get_offsets_for_marker(markerID)
        # Move to approach pose: same X,Y as handle in marker frame, but at standoff
        # distance on the marker Z axis.  This guarantees the final move to the handle
        # is purely along the marker's Z axis, avoiding collisions with the handle.
        approach_offset = np.array([handle_offset[0], handle_offset[1], approach_standoff])
        if not self._move_to_marker_offset(markerID, approach_offset):
            return False
        return self._move_to_marker_offset(markerID, handle_offset)

    def liftPlate(self, markerID=0):
        _, pickup_offset = self._get_offsets_for_marker(markerID)
        return self._move_to_marker_offset(markerID, pickup_offset)

    @_timed
    def pickupPlate(self, markerID=0):
        if not self.moveToMarker(markerID):
            self.get_logger().error(f"pickupPlate: moveToMarker failed for marker {markerID}.")
            return False
        self.close_gripper()
        time.sleep(3.0)
        if not self.liftPlate(markerID):
            self.get_logger().error(f"pickupPlate: liftPlate failed for marker {markerID}.")
            return False
        return True

    @_timed
    def placePlate(self, markerID=0):
        """Place a held build plate at the specified marker location."""
        handle_offset, _ = self._get_offsets_for_marker(markerID)
        # Move above destination
        if not self.liftPlate(markerID):
            self.get_logger().error(f"placePlate: liftPlate failed for marker {markerID}.")
            return False
        # Lower to handle position
        if not self._move_to_marker_offset(markerID, handle_offset):
            self.get_logger().error(f"placePlate: move to handle failed for marker {markerID}.")
            return False
        self.open_gripper()
        return True

    @_timed
    def transferPlate(self, source_id, dest_id, rescan_id, scan_distance=0.15):
        """
        Full plate-transfer sequence across three printers.

        1. Scan source marker  — retries at 0.85x and 0.70x if movement fails; aborts if all fail
        2. Pick up plate from source_id  — aborts on failure
        3. Place plate at dest_id        — aborts on failure
        4. Scan dest_id marker           — retries on movement failure (no abort)
        5. Scan rescan_id marker         — retries on movement failure; aborts if all fail
        6. Pick up plate from rescan_id  — aborts on failure
        7. Place plate back at source_id — aborts on failure
        """
        self.get_logger().info(
            f"transferPlate: source={source_id}, dest={dest_id}, rescan={rescan_id}"
        )

        # Step 1 – scan source; retry at closer distances only if movement fails
        self.get_logger().info(f"Step 1: scanning source marker {source_id}")
        move_ok, _ = self.scanToMarker(marker_id=source_id, viewing_distance=scan_distance)
        if not move_ok:
            move_ok, _ = self.scanToMarker(marker_id=source_id, viewing_distance=0.85 * scan_distance)
        if not move_ok:
            move_ok, _ = self.scanToMarker(marker_id=source_id, viewing_distance=0.70 * scan_distance)
        if not move_ok:
            self.get_logger().error(
                f"transferPlate: could not reach source marker {source_id}. Aborting."
            )
            return False

        # Step 2 – pick up from source
        self.get_logger().info(f"Step 2: picking up plate from marker {source_id}")
        _p = self._bambu_printers.get(source_id)
        if _p:
            _p.prepare_for_pickup()
        if not self.pickupPlate(markerID=source_id):
            self.get_logger().error(
                f"transferPlate: pickupPlate failed for marker {source_id}. Aborting."
            )
            return False

        # Step 3 – place at destination
        self.get_logger().info(f"Step 3: placing plate at marker {dest_id}")
        if not self.placePlate(markerID=dest_id):
            self.get_logger().error(
                f"transferPlate: placePlate failed for marker {dest_id}. Aborting."
            )
            return False

        # Step 4 – withdraw to approach standoff of destination marker
        self.get_logger().info(f"Step 4: withdrawing to approach standoff for marker {dest_id}")
        self._move_to_approach(dest_id)
        _p = self._bambu_printers.get(dest_id)
        if _p:
            _p.home()

        # Step 5 – scan rescan marker; retry at closer distances only if movement fails
        self.get_logger().info(f"Step 5: scanning marker {rescan_id} at {scan_distance} m")
        move_ok, _ = self.scanToMarker(marker_id=rescan_id, viewing_distance=scan_distance)
        if not move_ok:
            move_ok, _ = self.scanToMarker(marker_id=rescan_id, viewing_distance=0.85 * scan_distance)
        if not move_ok:
            move_ok, _ = self.scanToMarker(marker_id=rescan_id, viewing_distance=0.70 * scan_distance)
        if not move_ok:
            self.get_logger().error(
                f"transferPlate: could not reach rescan marker {rescan_id}. Aborting."
            )
            return False

        # Step 6 – pick up from rescan printer
        self.get_logger().info(f"Step 6: picking up plate from marker {rescan_id}")
        _p = self._bambu_printers.get(rescan_id)
        if _p:
            _p.prepare_for_pickup()
        if not self.pickupPlate(markerID=rescan_id):
            self.get_logger().error(
                f"transferPlate: pickupPlate failed for marker {rescan_id}. Aborting."
            )
            return False

        # Step 7 – place back at source
        self.get_logger().info(f"Step 7: placing plate at marker {source_id}")
        if not self.placePlate(markerID=source_id):
            self.get_logger().error(
                f"transferPlate: placePlate failed for marker {source_id}. Aborting."
            )
            return False

        # Withdraw to approach standoff of source marker
        self.get_logger().info(f"Withdrawing to approach standoff for marker {source_id}")
        self._move_to_approach(source_id)
        _p = self._bambu_printers.get(source_id)
        if _p:
            _p.home()

        self.get_logger().info("transferPlate: sequence complete.")
        self.open_gripper()
        time.sleep(1.0)
        return True

    @_timed
    def scanMarkerApproach(self, marker_id, viewing_distance=0.15):
        """
        Scan a marker at progressively closer distances.

        Starts at 1.75x the given viewing_distance and steps down to 1.0x.
        If the marker is not spotted at the longest distance (1.75x), the
        approach is aborted and the method returns False immediately.
        Returns True if the marker was seen at least once, False otherwise.
        """
        distances = [
            (1.75 * viewing_distance, 2.0),
            (1.50 * viewing_distance, 1.0),
            (1.25 * viewing_distance, 1.0),
            (1.00 * viewing_distance, 1.0),
            (1.00 * viewing_distance, 1.0),
        ]

        for i, (dist, pause) in enumerate(distances):
            move_ok, spotted = self.scanToMarker(marker_id=marker_id, viewing_distance=dist)
            time.sleep(pause)
            if i == 0 and not spotted:
                self.get_logger().warn(
                    f"scanMarkerApproach: marker {marker_id} not seen at max distance "
                    f"({dist:.3f} m) — aborting approach."
                )
                return False

        return True

    @_timed
    def scrapePlate(self, source_id, scrape_id, scan_distance=0.15, scrape_standoff=0.15, scrape_offset=None):
        """
        Pick up a plate from source_id, scrape it against the scrape_id marker surface,
        then return it to source_id.

        scrape_offset: 3-element [x, y, z] offset from the scrape marker origin in the
          marker's local frame at which the scrape is carried out.  Falls back to
          self.scrape_offset when not provided.

        1. Scan source marker  — retries at 0.85x and 0.70x if movement fails; aborts if all fail
        2. Pick up plate from source_id  — aborts on failure
        3. Scan scrape marker  — retries at 0.85x and 0.70x if movement fails; aborts if all fail
        4. Move to standoff position along the scrape marker's Z axis
        5. Move to scrape_offset position in the scrape marker's local frame
        6. Retract back to standoff along the scrape marker's Z axis
        7. Place plate back at source_id  — aborts on failure
        """
        if scrape_offset is None:
            scrape_offset = self.scrape_offset
        else:
            scrape_offset = np.array(scrape_offset, dtype=float)
        self.get_logger().info(
            f"scrapePlate: source={source_id}, scrape={scrape_id}, "
            f"standoff={scrape_standoff}, scrape_offset={scrape_offset}"
        )

        # Step 1 – scan source marker; retry at closer distances only if movement fails
        self.get_logger().info(f"Step 1: scanning source marker {source_id}")
        move_ok, _ = self.scanToMarker(marker_id=source_id, viewing_distance=scan_distance)
        if not move_ok:
            move_ok, _ = self.scanToMarker(marker_id=source_id, viewing_distance=0.85 * scan_distance)
        if not move_ok:
            move_ok, _ = self.scanToMarker(marker_id=source_id, viewing_distance=0.70 * scan_distance)
        if not move_ok:
            self.get_logger().error(
                f"scrapePlate: could not reach source marker {source_id}. Aborting."
            )
            return False

        # Step 2 – pick up plate from source
        self.get_logger().info(f"Step 2: picking up plate from marker {source_id}")
        if not self.pickupPlate(markerID=source_id):
            self.get_logger().error(
                f"scrapePlate: pickupPlate failed for marker {source_id}. Aborting."
            )
            return False
        # Freeze marker updates while scraping so the scrape surface marker pose is
        # not corrupted by the camera seeing it at close range during the approach.
        self.freeze_markers()

        '''# Step 3 – scan scrape marker; retry at closer distances only if movement fails
        self.get_logger().info(f"Step 3: scanning scrape marker {scrape_id}")
        move_ok, _ = self.scanToMarker(marker_id=scrape_id, viewing_distance=scan_distance)
        if not move_ok:
            move_ok, _ = self.scanToMarker(marker_id=scrape_id, viewing_distance=0.85 * scan_distance)
        if not move_ok:
            move_ok, _ = self.scanToMarker(marker_id=scrape_id, viewing_distance=0.70 * scan_distance)
        if not move_ok:
            self.get_logger().error(
                f"scrapePlate: could not reach scrape marker {scrape_id}. Aborting."
            )
            return False
            '''

        # Step 4 – move to standoff: same x, y as scrape_offset but at standoff Z distance,
        # so the final approach and retract are purely along the marker's Z axis.
        self.get_logger().info(f"Step 4: moving to scrape standoff (Z={scrape_standoff} m)")
        standoff_offset = np.array([scrape_offset[0], scrape_offset[1], scrape_standoff])
        if not self._move_to_marker_offset(scrape_id, standoff_offset):
            self.get_logger().error(
                f"scrapePlate: could not reach scrape standoff for marker {scrape_id}. Aborting."
            )
            return False

        # Step 5 – move to scrape_offset position in the marker's local frame
        self.get_logger().info(f"Step 5: moving to scrape offset {scrape_offset} in marker frame")
        if not self._move_to_marker_offset(scrape_id, scrape_offset):
            self.get_logger().error(
                f"scrapePlate: could not reach scrape depth for marker {scrape_id}. Aborting."
            )
            return False
        self.pause_timing()
        print(f"[scrapePlate] Plate at full scrape depth on marker {scrape_id}. Timing paused.")
        self.resume_timing()

        
        self.freeze_markers()
        # Step 6 – retract back along marker Z to standoff
        self.get_logger().info(f"Step 6: retracting to standoff (Z={scrape_standoff} m)")
        if not self._move_to_marker_offset(scrape_id, standoff_offset):
            self.get_logger().error(
                f"scrapePlate: retraction to standoff failed for marker {scrape_id}. Continuing."
            )

        # Step 7 – place plate back at source
        # Unfreeze so the camera can refresh the source marker pose before placing.
        self.unfreeze_markers()
        self.get_logger().info(f"Step 7: placing plate back at marker {source_id}")
        if not self.placePlate(markerID=source_id):
            self.get_logger().error(
                f"scrapePlate: placePlate failed for marker {source_id}. Aborting."
            )
            return False

        # Withdraw to approach standoff of source marker
        self.get_logger().info(f"Withdrawing to approach standoff for marker {source_id}")
        self._move_to_approach(source_id)

        self.get_logger().info("scrapePlate: sequence complete.")
        self.open_gripper()
        time.sleep(1.0)
        return True

    @_timed
    def go_home(self, velocity_scaling=0.2):
        """
        Move all joints to their zero (home) position.

        Because MoveIt plans from the actual current joint state reported by the
        hardware (Teensy encoder counts), this corrects any positional drift that
        accumulated from lost stepper steps during gripping or other stall events.
        A low velocity_scaling is used so the recovery move is slow and less
        likely to cause further stalls.
        """
        self.get_logger().warn(
            f"go_home: resyncing to home position (velocity_scaling={velocity_scaling}). "
            "Planning from actual encoder state to correct any step-loss drift."
        )
        # Brief settle so the robot is truly stationary before we read its pose.
        time.sleep(0.5)

        home_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        prev_velocity = self.moveit2.max_velocity
        prev_acceleration = self.moveit2.max_acceleration
        try:
            self.moveit2.max_velocity = velocity_scaling
            self.moveit2.max_acceleration = velocity_scaling
            self.freeze_markers()
            self.moveit2.move_to_configuration(joint_positions=home_joints)
            self.moveit2.wait_until_executed()
            time.sleep(self.move_settle_delay)
        finally:
            self.moveit2.max_velocity = prev_velocity
            self.moveit2.max_acceleration = prev_acceleration
            self.unfreeze_markers()

        self.get_logger().info("go_home: reached home position.")


