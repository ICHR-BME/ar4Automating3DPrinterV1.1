from ArucoDetector import ArucoDetectionViewer
import rclpy
import numpy as np
from pymoveit2 import GripperInterface
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped
import tf2_ros
from simulated3DPrinter import Simulated3DPrinter
import time
import threading


class printerAutomation(ArucoDetectionViewer):
    def __init__(self, calibration_mode=False, stream_source="webcam", camera_index=None, camera_keyword="GENERAL WEBCAM",
                 color_topic="/rgbd_camera/image", depth_topic="/rgbd_camera/depth_image", camera_info_topic="/rgbd_camera/camera_info",
                 feed_rotation_deg=0.0):
        super().__init__(source=stream_source,
                         camera_index=camera_index,
                         camera_keyword=camera_keyword,
                         color_topic=color_topic,
                         depth_topic=depth_topic,
                         camera_info_topic=camera_info_topic,
                         feed_rotation_deg=feed_rotation_deg)
        self.get_logger().info(f"printerAutomation initialized, calibration_mode={calibration_mode}")

        # Estimated marker frame name prefix
        self.estimatedMarkerPrefix = "estimated_marker_"

        # When True, open_gripper/close_gripper are no-ops (workaround for sim instability)
        self.gripper_disabled = False
        # When True, register_estimated_marker adds random noise to test scan robustness
        self.randomize_estimated_markers = False

        self.markerToHandleOffset = np.array([0.0, 0.05, 0.06])
        self.markerToPickupOffset = np.array([0.0, 0.20, 0.06])
        self.offsetOri = np.array([0.0, np.pi, np.pi / 2])

        # Gripper interface
        self.gripper = GripperInterface(
            node=self,
            gripper_joint_names=["gripper_jaw1_joint"],
            open_gripper_joint_positions=[0.012],
            closed_gripper_joint_positions=[0.030],
            gripper_group_name="ar_gripper",
            callback_group=self._cb_group,
            gripper_command_action_name="gripper_controller/gripper_cmd",
        )

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
        self.move_to_pose(goodPos, goodEuler)
        self.unfreeze_markers()
        return True

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
            bad_pos = bad_pos + random_dir * 0.05
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

    def scanToMarker(self, marker_id=0, viewing_distance=0.15):
        """Move the camera to face a known/estimated marker using its TF frame."""
        entry = self._find_marker_entry(marker_id)
        if entry is None:
            self.get_logger().error(f"Marker {marker_id} not found in found_markers. Register it first.")
            return False

        offsetPos = np.array([0.0, 0.0, viewing_distance])
        badPos, badEuler = self._apply_offset_in_marker_frame(
            entry['positionInBase'], entry['eulerInBase'], offsetPos, self.offsetOri,
        )

        goodPos, goodEuler = self.to_good_frame(badPos, badEuler)
        self.get_logger().info(f"Scanning marker {marker_id}: moving to viewing pos={goodPos}")
        self.freeze_markers()
        self.move_to_pose(goodPos, goodEuler)
        self.unfreeze_markers()

        # Pause to allow camera to observe the marker
        time.sleep(1.0)
        observed_entry = self._find_marker_entry(marker_id)
        if observed_entry is None or observed_entry.get('estimated', False):
            self.get_logger().warn(f"Marker {marker_id} was not observed by the camera after moving to view it.")
        return True

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

    def moveToMarker(self, markerID=0):
        self.open_gripper()
        self._move_to_marker_offset(markerID, self.markerToHandleOffset)

    def liftPlate(self, markerID=0):
        self._move_to_marker_offset(markerID, self.markerToPickupOffset)

    def pickupPlate(self, markerID=0):
        self.moveToMarker(markerID)
        self.close_gripper()
        time.sleep(3.0)
        self.liftPlate(markerID)

    def placePlate(self, markerID=0):
        """Place a held build plate at the specified marker location."""
        # Move above destination
        self.liftPlate(markerID)
        # Lower to handle position
        self._move_to_marker_offset(markerID, self.markerToHandleOffset)
        # Release
        self.open_gripper()
        time.sleep(1.0)

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
    print(" 11) Place plate at marker")
    print("  4) Move to pose (manual)")
    print("  5) List detected markers")
    print("  6) Print current end-effector pose")
    print("  7) Set marker offsets")
    print("  8) Toggle marker updates")
    print("  9) Scan to marker (by ID, uses TF)")
    print(" 10) Go home & resync (correct step-loss drift)")
    print("  0) Quit")
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
            pos = _parse_floats("  Enter target pos (x y z): ", 3)
            if pos is None:
                continue
            orient = _parse_floats("  Enter target orient (roll pitch yaw): ", 3)
            if orient is None:
                orient = [0.0, 0.0, 0.0]
            node.get_logger().info(f"User requested move_to_pose({pos}, {orient})")
            node.move_to_pose(np.array(pos), np.array(orient))

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
            if hasattr(node, 'pose') and node.pose is not None:
                print(f"  Current EEF pose: {node.pose}")
            else:
                print("  Pose not yet available (waiting for joint_states).")

        elif choice == "7":
            print(f"  Current handle offset:  {node.markerToHandleOffset}")
            print(f"  Current pickup offset:  {node.markerToPickupOffset}")
            which = input("  Edit (h)andle offset, (p)ickup offset, or (b)oth? ").strip().lower()
            if which in ('h', 'b'):
                val = _parse_floats("  New handle offset (x y z): ", 3)
                if val:
                    node.markerToHandleOffset = np.array(val)
                    print(f"  Handle offset set to {node.markerToHandleOffset}")
            if which in ('p', 'b'):
                val = _parse_floats("  New pickup offset (x y z): ", 3)
                if val:
                    node.markerToPickupOffset = np.array(val)
                    print(f"  Pickup offset set to {node.markerToPickupOffset}")
            if which not in ('h', 'p', 'b'):
                print("  Unknown selection.")

        elif choice == "8":
            if node.stream.marker_updates_enabled:
                node.freeze_markers()
                print("  Marker updates FROZEN.")
            else:
                node.unfreeze_markers()
                print("  Marker updates RESUMED.")

        elif choice == "9":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            dist = _parse_floats("  Viewing distance [0.15]: ", 1)
            dist = dist[0] if dist else 0.15
            node.get_logger().info(f"User requested scanToMarker({mid}, dist={dist})")
            node.scanToMarker(marker_id=mid, viewing_distance=dist)

        elif choice == "10":
            scale = _parse_floats("  Velocity scaling [0.2]: ", 1)
            scale = scale[0] if scale else 0.2
            node.get_logger().info(f"User requested go_home(velocity_scaling={scale})")
            node.go_home(velocity_scaling=scale)

        elif choice == "11":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            node.get_logger().info(f"User requested placePlate({mid})")
            node.placePlate(markerID=mid)

        elif choice == "0":
            print("  Shutting down...")
            rclpy.shutdown()
            break

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

        # Source printer: marker ID 0 (left side)
        printer_source = Simulated3DPrinter(
            node=node,
            pos=[0.37, -0.2, 0.21],
            orient=[0.0, 0.1, np.pi],
            door_marker_texture='materials/textures/marker6x6_0.png',
        )
        printer_source.spawn_fast()

        # Destination printer: marker ID 1 (right side)
        printer_dest = Simulated3DPrinter(
            node=node,
            pos=[0.37, 0.2, 0.21],
            orient=[0.0, 0.0, 0.0],
            door_marker_texture='materials/textures/marker6x6_1.png',
        )
        printer_dest.spawn_fast()

    else:
        stream_source = "webcam"  # Use webcam for real environment
        node = printerAutomation(calibration_mode=False,stream_source=stream_source, feed_rotation_deg=90.0)
        node.stream.distance_scale = 1.0/0.702  # Correct webcam distance underestimation (~50%)

        # Single physical printer
        printer = Simulated3DPrinter(
            node=node,
            pos=[0.37, -0.17, 0.21],
            orient=[0.0, 0.0, np.pi],
        )

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
    for _ in range(50):  # up to 5 seconds
        if hasattr(node, 'pose') and node.pose is not None:
            break
        time.sleep(0.1)

    # Run the initial scan (blocks until move_to_pose completes)
    node.get_logger().info("Starting initial scan for markers...")
    if runVirtual:
        # Register both printer door markers using their known spawn poses
        bad_pos, bad_euler = printer_source.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=0, bad_pos=bad_pos, bad_euler=bad_euler)

        bad_pos, bad_euler = printer_dest.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=1, bad_pos=bad_pos, bad_euler=bad_euler)

        # Scan to source marker so the camera gets a real detection
        node.get_logger().info("Scanning source printer (marker 0)...")
        node.scanToMarker(marker_id=0, viewing_distance=0.20)

        # Scan to destination marker so the camera gets a real detection
        node.get_logger().info("Scanning destination printer (marker 1)...")
        node.scanToMarker(marker_id=1, viewing_distance=0.20)

        # Pick up the plate from the source printer and place it on the destination
        node.get_logger().info("Picking up plate from source printer (marker 0)...")
        node.pickupPlate(markerID=0)

        node.get_logger().info("Placing plate on destination printer (marker 1)...")
        node.placePlate(markerID=1)

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
        bad_pos, bad_euler = printer.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=0, bad_pos=bad_pos, bad_euler=bad_euler)
        # Move camera to view the marker from 15 cm away
        node.scanToMarker(marker_id=0, viewing_distance=0.20)

        # Pick up the plate and place it back at the same marker
        #node.pickupPlate(markerID=0)
        #node.placePlate(markerID=0)

    node.get_logger().info("Initial scan complete.")

    # Now start the interactive input loop on the main thread
    _input_thread(node)



if __name__ == '__main__':
    main()