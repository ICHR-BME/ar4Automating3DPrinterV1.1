from ArucoDetector import ArucoDetectionViewer
import rclpy
import numpy as np
import os
import json
from pymoveit2 import GripperInterface
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
import tf2_ros
from simulated3DPrinter import Simulated3DPrinter
import time
import threading


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class printerAutomation(ArucoDetectionViewer):
    def __init__(self, calibration_mode=False, stream_source="webcam", camera_index=None, camera_keyword="GENERAL WEBCAM",
                 color_topic="/rgbd_camera/image", depth_topic="/rgbd_camera/depth_image", camera_info_topic="/rgbd_camera/camera_info",
                 feed_rotation_deg=0.0, marker_sizes=None):
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
                'handleOffset': np.array([0.0, 0.07, 0.09]),
                'pickupOffset': np.array([0.0, 0.175, 0.09]),
            },
            # Printer with the marker to the side
            'box_offset': {
                'handleOffset': np.array([0.0, 0.045, 0.105]),
                'pickupOffset': np.array([0.0, 0.145, 0.105]),
            },
        }
        ## Map marker_id -> config name. IDs not listed fall back to 'box_offset'.
        self.marker_offset_config = {}
        
        self.offsetOri = np.array([0.0, np.pi, np.pi / 2])

        # The camera is mounted below the gripper. Raise the end effector by this
        # amount (metres, base-link Z) when scanning so the camera aligns with the marker.
        self.camera_z_offset = 0.06

        # Persistent state save file — written every 5 s and loaded at startup
        self._state_save_path = os.path.join(_SCRIPT_DIR, "printer_state.json")
        self.create_timer(5.0, self._auto_save_state)

        # Gripper interface
        self.gripper = GripperInterface(
            node=self,
            gripper_joint_names=["gripper_jaw1_joint"],
            open_gripper_joint_positions=[0.00],
            closed_gripper_joint_positions=[0.014],
            gripper_group_name="ar_gripper",
            callback_group=self._cb_group,
            gripper_command_action_name="gripper_controller/gripper_cmd",
        )

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

        # Restore marker poses as estimated entries
        for m in data.get("markers", []):
            marker_id = int(m["id"])
            pos = np.array(m["positionInBase"], dtype=float)
            euler = np.array(m["eulerInBase"], dtype=float)
            self.register_estimated_marker(marker_id=marker_id, bad_pos=pos, bad_euler=euler)

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

        # Pause to allow camera to observe the marker
        time.sleep(1.0)
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
        self.moveToMarker(markerID=source_id)
        time.sleep(2)
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

        # Step 4 – re-observe destination marker; retry on movement failure (no abort)
        self.get_logger().info(f"Step 4: scanning marker {dest_id} at {scan_distance} m")
        move_ok, _ = self.scanToMarker(marker_id=dest_id, viewing_distance=scan_distance)
        if not move_ok:
            move_ok, _ = self.scanToMarker(marker_id=dest_id, viewing_distance=0.85 * scan_distance)
        if not move_ok:
            self.scanToMarker(marker_id=dest_id, viewing_distance=0.70 * scan_distance)

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

        # Final scan of source marker; retry on movement failure (no abort)
        move_ok, _ = self.scanToMarker(marker_id=source_id, viewing_distance=scan_distance)
        if not move_ok:
            move_ok, _ = self.scanToMarker(marker_id=source_id, viewing_distance=0.85 * scan_distance)
        if not move_ok:
            self.scanToMarker(marker_id=source_id, viewing_distance=0.70 * scan_distance)

        self.get_logger().info("transferPlate: sequence complete.")
        self.open_gripper()
        time.sleep(1.0)
        return True

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


def _print_menu():
    """Print the interactive command menu."""
    print("\n" + "=" * 50)
    print("  3D Printer Automation - Command Menu")
    print("=" * 50)
    print("  1) Scan location for markers (manual pos/orient)")
    print("  2) Move to marker")
    print("  3) Pickup plate (move + lift)")
    print("  4) Place plate at marker")
    print("  5) List detected markers")
    print("  6) Scan to marker (by ID, uses TF)")
    print("  7) Go home & resync (correct step-loss drift)")
    print("  8) Transfer plate (source → dest, rescan → place)")
    print("=" * 50)


def _parse_floats(prompt, count=None):
    """Prompt for space-separated floats. Returns list of floats or None on error."""
    try:
        raw = input(prompt).strip()
        values = [float(v) for v in raw.split()]
        if count is not None and len(values) != count:
            print(f"  Expected {count} values, got {len(values)}")
            return None
        return values
    except ValueError:
        print("  Invalid input. Enter space-separated numbers.")
        return None


def _input_thread(node):
    """
    Runs in a background thread. Reads user commands from stdin
    and dispatches them on the node (which is spinning in the main thread).
    """
    # Wait for the system to initialize
    time.sleep(5.0)
    print("\n[INFO] System ready. Type a command number.")

    while rclpy.ok():
        _print_menu()
        try:
            choice = input(">> ").strip()
        except EOFError:
            break

        # ROS log messages can interleave with terminal input, causing extra
        # characters to be buffered before the intended option digit(s).
        # e.g. user types "1" then a log line prints, then they type "9" → "19"
        # If the choice is unrecognised, try stripping one leading character.
        _valid_choices = {"1", "2", "3", "4", "5", "6", "7", "8"}
        if choice not in _valid_choices and len(choice) >= 2 and choice[1:] in _valid_choices:
            choice = choice[1:]

        if choice == "1":
            pos = _parse_floats("  Enter estimated pos (x y z): ", 3)
            if pos is None:
                continue
            orient = _parse_floats("  Enter estimated orient (roll pitch yaw) [0 0 0]: ", 3)
            if orient is None:
                orient = [0.0, 0.0, 0.0]
            dist = _parse_floats("  Viewing distance [0.15]: ", 1)
            dist = dist[0] if dist else 0.15
            node.get_logger().info(f"User requested scanLocationForMarkers at {pos}")
            node.scanLocationForMarkers(
                estimated_pos=pos,
                estimated_orient=orient,
                viewing_distance=dist,
            )

        elif choice == "2":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            node.get_logger().info(f"User requested moveToMarker({mid})")
            node.moveToMarker(markerID=mid)

        elif choice == "3":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            node.get_logger().info(f"User requested pickupPlate({mid})")
            node.pickupPlate(markerID=mid)

        elif choice == "4":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            node.get_logger().info(f"User requested placePlate({mid})")
            node.placePlate(markerID=mid)

        elif choice == "5":
            markers = node.marker_poses
            if markers:
                print(f"\n  Found {len(markers)} marker(s):")
                for entry in markers:
                    pos = entry.get('positionInWorld', 'N/A')
                    ori = entry.get('orientInWorld', 'N/A')
                    print(f"    ID {entry['id']}: pos={pos}, orient={ori}")
            else:
                print("  No markers found yet.")

        elif choice == "6":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            dist = _parse_floats("  Viewing distance [0.15]: ", 1)
            dist = dist[0] if dist else 0.15
            node.get_logger().info(f"User requested scanToMarker({mid}, dist={dist})")
            node.scanToMarker(marker_id=mid, viewing_distance=dist)

        elif choice == "7":
            scale = _parse_floats("  Velocity scaling [0.2]: ", 1)
            scale = scale[0] if scale else 0.2
            node.get_logger().info(f"User requested go_home(velocity_scaling={scale})")
            node.go_home(velocity_scaling=scale)

        elif choice == "8":
            ids = _parse_floats("  Source, dest, rescan marker IDs (e.g. 0 1 2): ", 3)
            if ids is None:
                continue
            source_id, dest_id, rescan_id = int(ids[0]), int(ids[1]), int(ids[2])
            dist = _parse_floats("  Scan distance [0.15]: ", 1)
            dist = dist[0] if dist else 0.15
            node.get_logger().info(
                f"User requested transferPlate({source_id}, {dest_id}, {rescan_id}, scan_distance={dist})"
            )
            node.transferPlate(source_id=source_id, dest_id=dest_id, rescan_id=rescan_id, scan_distance=dist)

        else:
            print("  Unknown option. Try again.")


def main():
    rclpy.init()
    runVirtual = 0

    if runVirtual:
        stream_source = "ros"  # Use ROS topic stream for simulated environment
        node = printerAutomation(calibration_mode=False,stream_source=stream_source)
        node.gripper_disabled = True  # Workaround: gripper causes joint drift in simulation
        node.randomize_estimated_markers = True

        # Destination printer: marker ID 0
        printer1 = Simulated3DPrinter(
            node=node,
            pos=[0.22, -0.2, 0.21],
            orient=[0.0, 0.0, np.pi],
            door_marker_texture='materials/textures/marker6x6_0.png',
        )
        printer1.spawn_fast()

        # Source printer: marker ID 1
        printer2 = Simulated3DPrinter(
            node=node,
            pos=[0.44, -0.2, 0.21],
            orient=[0.0, 0.1, np.pi],
            door_marker_texture='materials/textures/marker6x6_1.png',
        )
        printer2.spawn_fast()

        

        printer3 = Simulated3DPrinter(
            node=node,
            pos=[0.60, 0.1, 0.21],
            orient=[0.0, 0.0, 3/2*np.pi],
            door_marker_texture='materials/textures/marker6x6_2.png',
        )
        printer3.spawn_fast()

    else:
        stream_source = "webcam"  # Use webcam for real environment
        node = printerAutomation(calibration_mode=False,stream_source=stream_source, feed_rotation_deg=90.0,marker_sizes=[0.03, 0.025])
        node.stream.distance_scale = 1.0/0.702  # Correct webcam distance underestimation (~50%)

        # Destination printer: marker ID 0
        # No spawn_fast() on physical robot – that calls ros_gz_sim create which blocks
        # indefinitely when Gazebo is not running, preventing the executor from ever starting.
        # get_door_marker_pose_in_base() is pure geometry; it doesn't need Gazebo.
        printer1 = Simulated3DPrinter(
            node=node,
            pos=[0.26, -0.3, 0.07],
            orient=[0.0, 0.0, np.pi],
            door_marker_texture='materials/textures/marker6x6_0.png',
        )

        # Source printer: marker ID 1
        printer2 = Simulated3DPrinter(
            node=node,
            pos=[0.48, -0.3, 0.07],
            orient=[0.0, 0.0, np.pi],
            door_marker_texture='materials/textures/marker6x6_1.png',
        )

        printer3 = Simulated3DPrinter(
            node=node,
            pos=[0.65, 0.1, 0.07],
            orient=[0.0, 0.0, 3/2*np.pi],
            door_marker_texture='materials/textures/marker6x6_2.png',
        )
        '''printer = Simulated3DPrinter(
            node=node,
            pos=[0.37, -0.17, 0.16],
            orient=[0.0, 0.0, np.pi],
        )'''
        '''printer1 = Simulated3DPrinter(
            node=node,
            pos=[0.33, -0.1, 0.02],
            orient=[0.3, 0.0, np.pi],
        )
        printer2 = Simulated3DPrinter(
            node=node,
            pos=[0.60, 0.05, 0.07],
            orient=[0.0, 0.0, 3*np.pi/2],
        )'''
    # Spin both the ROS node and the stream's internal ROS node
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    # If the stream provides a ROS Node (source="ros"), add it to the executor.
    # If using a webcam (no _ros_node), run the stream.run loop in a background thread
    if hasattr(node.stream, "_ros_node"):
        executor.add_node(node.stream._ros_node)
    else:
        stream_thread = threading.Thread(target=node.stream.run, daemon=True)
        stream_thread.start()

    # Start the executor in a background thread so TF / joint_states / callbacks work
    def _resilient_spin(executor):
        """Keep spinning even if individual callbacks raise exceptions."""
        while rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                # Log but don't crash the spin thread
                node.get_logger().warn(f"Executor spin error (recovering): {e}")
                time.sleep(0.01)

    spin_thread = threading.Thread(target=_resilient_spin, args=(executor,), daemon=True)
    spin_thread.start()

    # Wait for joint_states to arrive before scanning
    node.get_logger().info("Waiting for joint_states before initial scan...")
    for _ in range(100):  # up to 10 seconds
        if node._last_joint_msg is not None:
            break
        time.sleep(0.1)
    else:
        node.get_logger().warn("Timed out waiting for joint_states - proceeding anyway")

    # Run the initial scan (blocks until move_to_pose completes)
    node.get_logger().info("Starting initial scan for markers...")
    node.load_state()
    if runVirtual:
        # Register both printer door markers using their known spawn poses
        bad_pos, bad_euler = printer1.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=0, bad_pos=bad_pos, bad_euler=bad_euler)

        bad_pos, bad_euler = printer2.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=1, bad_pos=bad_pos, bad_euler=bad_euler)


        bad_pos, bad_euler = printer3.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=2, bad_pos=bad_pos, bad_euler=bad_euler)

        # Scan to source marker so the camera gets a real detection
        #node.get_logger().info("Scanning source printer (marker 0)...")
        #node.scanToMarker(marker_id=0, viewing_distance=0.20)

        # Scan to destination marker so the camera gets a real detection
        #node.get_logger().info("Scanning destination printer (marker 1)...")
        #node.scanToMarker(marker_id=1, viewing_distance=0.20)

        # Pick up the plate from the source printer and place it on the destination
        #node.get_logger().info("Picking up plate from source printer (marker 0)...")
        #node.pickupPlate(markerID=0)

        #node.get_logger().info("Placing plate on destination printer (marker 1)...")
        #node.placePlate(markerID=1)

    else:
        # For the physical setup, provide a rough estimate of where the marker is.
        # register_estimated_marker takes base_link (bad frame) coordinates.
        # These will be overwritten once the camera detects the real marker.
        '''est_pos, est_euler = node.to_bad_frame(
            np.array([0.29, 0.15, 0.16]),   # estimated good-frame position
            np.array([0.0, 0.0, 0.0]),       # estimated good-frame orientation
        )
        node.register_estimated_marker(marker_id=0, bad_pos=est_pos, bad_euler=est_euler)
        node.scanToMarker(marker_id=0, viewing_distance=0.0)'''
        viewing_distance = 0.15
        node.marker_offset_config[0] = 'box_offset'
        node.marker_offset_config[1] = 'box_offset'
        node.marker_offset_config[2] = 'printer_offset'

        bad_pos, bad_euler = printer1.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=0, bad_pos=bad_pos, bad_euler=bad_euler)
        bad_pos, bad_euler = printer2.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=1, bad_pos=bad_pos, bad_euler=bad_euler)

        bad_pos, bad_euler = printer3.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=2, bad_pos=bad_pos, bad_euler=bad_euler)

        # Record printer configs so they are persisted in every save
        node.register_printers([
            {"marker_id": 0, "pos": [0.26, -0.3, 0.07], "orient": [0.0, 0.0, np.pi],
             "door_marker_texture": 'materials/textures/marker6x6_0.png'},
            {"marker_id": 1, "pos": [0.48, -0.3, 0.07], "orient": [0.0, 0.0, np.pi],
             "door_marker_texture": 'materials/textures/marker6x6_1.png'},
            {"marker_id": 2, "pos": [0.65, 0.1, 0.07],  "orient": [0.0, 0.0, 3/2*np.pi],
             "door_marker_texture": 'materials/textures/marker6x6_2.png'},
        ])

        
        # View markers 0, 1, 2 — abort approach if not seen at max distance
        node.scanMarkerApproach(marker_id=0, viewing_distance=viewing_distance)
        node.scanMarkerApproach(marker_id=1, viewing_distance=viewing_distance)
        node.scanMarkerApproach(marker_id=2, viewing_distance=viewing_distance)
        
        
        
        
        
        # Pick up the plate and place it back at the same marker
        #node.pickupPlate(markerID=0)
        #node.placePlate(markerID=0)

    node.get_logger().info("Initial scan complete.")

    # Now start the interactive input loop on the main thread
    _input_thread(node)



if __name__ == '__main__':
    main()