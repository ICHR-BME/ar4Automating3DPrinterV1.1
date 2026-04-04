from ArucoDetector import ArucoDetectionViewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
from poseReader import PoseReader
import numpy as np
from pymoveit2 import GripperInterface
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, TransformStamped, PoseStamped
import tf2_ros
from rclpy.time import Time
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

        if not hasattr(self, 'tf2_broadcaster'):
            self.tf2_broadcaster = tf2_ros.TransformBroadcaster(self)

        #self.markerToHandleOffset = np.array([0.0, -0.025, 0.033])
        #self.markerToPickupOffset = np.array([0.0, 0.11, 0.033])
        self.markerToHandleOffset = np.array([0.0, -0.025, 0.033])
        self.markerToPickupOffset = np.array([0.0, 0.11, 0.033])
        self.offsetOri = np.array([0.0, np.pi, np.pi / 2])

        # Gripper interface
        self.gripper = GripperInterface(
            node=self,
            gripper_joint_names=["gripper_jaw1_joint"],
            open_gripper_joint_positions=[0.012],
            closed_gripper_joint_positions=[0.025],
            gripper_group_name="ar_gripper",
            callback_group=self._cb_group,
            gripper_command_action_name="gripper_controller/gripper_cmd",
        )

    # NO detected_markers property needed — use self.marker_poses inherited
    # from ArucoDetectionViewer which reads self.stream.found_markers

    def open_gripper(self):
        if self.gripper_disabled:
            self.get_logger().info("Gripper disabled — skipping open.")
            return
        self.get_logger().info("Opening gripper...")
        self.gripper.open()
        #self.gripper.wait_until_executed()

    def close_gripper(self):
        if self.gripper_disabled:
            self.get_logger().info("Gripper disabled — skipping close.")
            return
        self.get_logger().info("Closing gripper...")
        self.gripper.close()
        #self.gripper.wait_until_executed()

    def freeze_markers(self):
        """Disable marker pose updates. Call before moving the robot."""
        self.stream.marker_updates_enabled = False
        self.get_logger().info("Marker pose updates frozen.")

    def unfreeze_markers(self):
        """Re-enable marker pose updates. Call after the robot has stopped."""
        self.stream.marker_updates_enabled = True
        self.get_logger().info("Marker pose updates resumed.")

    def register_estimated_marker(self, marker_id, bad_pos, bad_euler):
        """
        Pre-populate an estimated marker pose in both the TF tree and found_markers.
        
        This broadcasts an ``aruco_marker_{id}`` frame in base_link and stores
        a synthetic entry in ``stream.found_markers`` so that scanToMarker /
        moveToMarker can work immediately.
        
        Once the camera actually detects this marker, _enrich_marker_pose will
        overwrite both the TF frame and the found_markers entry with the real
        measurement.
        
        Parameters:
            marker_id: ArUco marker ID (int)
            bad_pos:   np.array([x, y, z]) in base_link
            bad_euler: np.array([roll, pitch, yaw]) intrinsic XYZ in base_link
        """
        bad_pos = np.array(bad_pos, dtype=float)
        bad_euler = np.array(bad_euler, dtype=float)
        # Add a random offset of magnitude 0.05 to the estimated marker position
        rng = np.random.default_rng()
        random_dir = rng.normal(size=3)
        random_dir /= np.linalg.norm(random_dir)
        random_offset = random_dir * 0.05
        bad_pos = bad_pos + random_offset

        # Add a random orientation offset of magnitude 0.05 radians
        random_ori_dir = rng.normal(size=3)
        random_ori_dir /= np.linalg.norm(random_ori_dir)
        random_ori_offset = random_ori_dir * 0.05
        bad_euler = bad_euler + random_ori_offset
        tf2Name = f"{self.markerNamePrefix}{marker_id}"

        # Broadcast as a static transform so it persists in the TF buffer
        # regardless of executor timing (static transforms use latched QoS).
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = tf2Name
        t.transform.translation.x = float(bad_pos[0])
        t.transform.translation.y = float(bad_pos[1])
        t.transform.translation.z = float(bad_pos[2])
        q = R.from_euler("XYZ", bad_euler, degrees=False).as_quat()
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])
        self.tf2_static_broadcaster.sendTransform(t)

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
            # Dummy camera-frame keys so the HUD panel and enrich_fn don't crash
            'positionFromCamera': np.array([0.0, 0.0, 0.0]),
            'eulerFromCamera': np.array([0.0, 0.0, 0.0]),
            'orientFromCamera': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            'distanceFromCamera': 0.0,
            'estimated': True,  # flag so callers can tell this is not camera-detected
        }
        self.stream.found_markers[marker_id] = entry
        self.get_logger().info(
            f"Registered estimated marker {marker_id} at base_link pos={bad_pos}, euler={bad_euler}"
        )

    def scanToMarker(self, marker_id=0, viewing_distance=0.15):
        """
        Move the camera to face a known/estimated marker using its TF frame.
        
        Works with both estimated markers (from register_estimated_marker) and
        already-detected markers.  The marker must have an entry in
        ``stream.found_markers`` so that its TF frame ``aruco_marker_{id}``
        exists.
        
        Parameters:
            marker_id: ArUco marker ID whose TF frame to look up
            viewing_distance: How far from the marker face to position the camera (m)
        """
        tf2Name = f"{self.markerNamePrefix}{marker_id}"
        markers = self.marker_poses
        entry = None
        for m in markers:
            if m['id'] == marker_id:
                entry = m
                break
        if entry is None:
            self.get_logger().error(f"Marker {marker_id} not found in found_markers. Register it first.")
            return False

        bad_pos = entry['positionInBase']
        bad_euler = entry['eulerInBase']

        offsetPos = np.array([0.0, 0.0, viewing_distance])
        offsetOri = self.offsetOri

        # Re-broadcast and look up TF
        badPos, badEuler = None, None
        for attempt in range(20):
            self.broadcast_marker_transform(bad_pos, bad_euler, child_frame=tf2Name)
            time.sleep(0.05)
            try:
                badPos, badEuler = self.applyFrameChange(
                    offsetPos, offsetOri,
                    source_frame="base_link", target_frame=tf2Name,
                )
                if badPos is not None:
                    break
            except Exception as e:
                self.get_logger().warn(f"TF lookup attempt {attempt+1}/20 for '{tf2Name}' failed: {e}")

        if badPos is None:
            self.get_logger().error(f"Failed to resolve TF frame '{tf2Name}' after 20 attempts")
            return False

        goodPos, goodEuler = self.to_good_frame(badPos, badEuler)

        self.get_logger().info(f"Scanning marker {marker_id}: moving to viewing pos={goodPos}")
        # Only allow marker updates while observing the marker
        self.freeze_markers()
        self.move_to_pose(goodPos, goodEuler)
        self.unfreeze_markers()
        # Pause to allow camera to observe the marker
        time.sleep(1.0)
        # Check if marker is still estimated
        markers = self.marker_poses
        observed = False
        for m in markers:
            if m['id'] == marker_id and not m.get('estimated', False):
                observed = True
                break
        if not observed:
            self.get_logger().warn(f"Marker {marker_id} was not observed by the camera after moving to view it.")
        self.freeze_markers()
        return True


    def scanLocationForMarkers(self, estimated_pos, estimated_orient=[0,0,0], viewing_distance=0.15, frame_name=None):
        """
        Move the camera to face an estimated marker location.
        
        This creates a temporary TF frame at the estimated location and moves
        the robot to view it, similar to moveToMarker but for positions where
        we expect to find a marker.
        
        Parameters:
            estimated_pos: [x, y, z] estimated marker position in base_link frame
            estimated_orient: [roll, pitch, yaw] estimated marker orientation (radians).
                             If None, assumes marker faces toward robot origin.
            viewing_distance: Distance from marker to position the camera (meters)
            frame_name: Optional custom frame name. If None, uses "estimated_marker_0"
        """
        estimated_pos = np.array(estimated_pos)        
        
        # Create frame name
        if frame_name is None:
            frame_name = f"{self.estimatedMarkerPrefix}0"
        
        # Define offset from marker (position camera at viewing_distance in front, facing the marker)
        offsetPos = np.array([0.0, 0.0, viewing_distance])
        offsetOri = np.array([0.0, 0.0, 0.0])
        
        # Transform offset from base_link to the estimated marker frame with retry
        markerBadPos, markerBadEuler = self.to_bad_frame(estimated_pos, estimated_orient)

        # Ensure the TF broadcaster exists before the retry loop
        if not hasattr(self, 'tf2_broadcaster'):
            self.tf2_broadcaster = tf2_ros.TransformBroadcaster(self)

        badPos = None
        badEuler = None
        for attempt in range(20):
            # Broadcast the estimated marker frame
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "base_link"
            t.child_frame_id = frame_name
            t.transform.translation.x = float(markerBadPos[0])
            t.transform.translation.y = float(markerBadPos[1])
            t.transform.translation.z = float(markerBadPos[2])
            q = R.from_euler("XYZ", markerBadEuler, degrees=False).as_quat()
            t.transform.rotation.x = float(q[0])
            t.transform.rotation.y = float(q[1])
            t.transform.rotation.z = float(q[2])
            t.transform.rotation.w = float(q[3])
            self.tf2_broadcaster.sendTransform(t)
            
            # Give the TF buffer time to receive the broadcast via the executor
            time.sleep(0.2)
            
            try:
                badPos, badEuler = self.applyFrameChange(
                    offsetPos, offsetOri,
                    source_frame="base_link", target_frame=frame_name
                )
                if badPos is not None:
                    self.get_logger().info(f"TF lookup succeeded on attempt {attempt+1}")
                    break
            except Exception as e:
                self.get_logger().warn(f"Attempt {attempt+1}/20: TF lookup for '{frame_name}' failed: {e}")
        
        if badPos is None:
            self.get_logger().error(f"Failed to get transform to {frame_name} after 20 attempts")
            return False
        
        goodPos, goodEuler = self.to_good_frame(badPos, badEuler)
        
        self.get_logger().info(f'Scanning for markers at estimated position: {estimated_pos}')
        self.get_logger().info(f'Moving to viewing position: {goodPos}')
        
        self.freeze_markers()
        self.move_to_pose(goodPos, goodEuler)
        self.unfreeze_markers()
        return True
    
    def scanMultipleLocations(self, locations, viewing_distance=0.15, pause_duration=2.0):
        """
        Scan multiple estimated marker locations sequentially.
        
        Parameters:
            locations: List of [x, y, z] positions or tuples of (pos, orient)
            viewing_distance: Distance from each marker to position the camera
            pause_duration: Time to pause at each location for marker detection (seconds)
        """
        for i, location in enumerate(locations):
            # Handle both position-only and (position, orientation) formats
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

                markers = self.marker_poses  # <-- reads from stream.found_markers
                if markers:
                    self.get_logger().info(f"Detected {len(markers)} markers at location {i+1}")
                else:
                    self.get_logger().info(f"No markers detected at location {i+1}")

    def pickupPlate(self, markerID=0):
        self.moveToMarker(markerID)
        self.close_gripper()
        time.sleep(3.0)
        self.liftPlate(markerID)

    def placePlate(self, markerID=0):
        """
        Place a held build plate at the specified marker location.

        Mirrors pickupPlate in reverse: move above the destination using the
        pickup offset, lower to the handle offset (without releasing the gripper),
        open the gripper to release the plate, then lift back up.

        Parameters:
            markerID: ArUco marker ID at the destination location
        """
        # Step 1: move above destination (same motion as liftPlate)
        self.liftPlate(markerID)

        # Step 2: lower to handle position without touching the gripper
        offsetPos = self.markerToHandleOffset
        offsetOri = self.offsetOri
        markers = self.marker_poses
        if markers:
            for entry in markers:
                if entry['id'] == markerID:
                    bad_pos = entry['positionInBase']
                    bad_euler = entry['eulerInBase']
                    tf2Name = entry['tf2Name']

                    badPos, badEuler = None, None
                    for attempt in range(20):
                        self.broadcast_marker_transform(bad_pos, bad_euler, child_frame=tf2Name)
                        time.sleep(0.05)
                        try:
                            badPos, badEuler = self.applyFrameChange(offsetPos, offsetOri, source_frame="base_link", target_frame=tf2Name)
                            break
                        except Exception as e:
                            self.get_logger().warn(f"TF lookup attempt {attempt+1}/20 for '{tf2Name}' failed: {e}")
                    if badPos is None:
                        self.get_logger().error(f"Could not resolve TF frame '{tf2Name}' after 20 attempts")
                        return
                    goodPos, goodEuler = self.to_good_frame(badPos, badEuler)

                    self.get_logger().info(f'Placing plate at marker ID {markerID}: lowering to handle pos={bad_pos}')
                    self.freeze_markers()
                    self.move_to_pose(goodPos, goodEuler)
                    self.unfreeze_markers()

                    # Step 3: release
                    self.open_gripper()
                    time.sleep(1.0)

                    return

            self.get_logger().warn(f"Marker ID {markerID} not found in detected marker poses.")
        else:
            self.get_logger().warn("No marker poses available to place plate.")

    def moveToMarker(self, markerID=0):
        self.open_gripper()
        foundMarker = 0
        print(f"Moving to marker ID {markerID}...")
        offsetPos = self.markerToHandleOffset
        offsetOri = self.offsetOri
        markers = self.marker_poses
        if markers:
            for entry in markers:
                if entry['id'] == markerID:
                    is_estimated = entry.get('estimated', False)
                    bad_pos = entry['positionInBase']
                    bad_euler = entry['eulerInBase']
                    tf2Name = entry['tf2Name']
                    self.get_logger().warn(
                        f"[moveToMarker] ID={markerID} "
                        f"estimated={is_estimated} "
                        f"positionInBase={np.round(bad_pos, 4)}"
                    )

                    # Re-broadcast marker TF and poll until the frame is available
                    badPos, badEuler = None, None
                    for attempt in range(20):
                        self.broadcast_marker_transform(bad_pos, bad_euler, child_frame=tf2Name)
                        time.sleep(0.05)
                        try:
                            badPos, badEuler = self.applyFrameChange(offsetPos, offsetOri, source_frame="base_link", target_frame=tf2Name)
                            break
                        except Exception as e:
                            self.get_logger().warn(f"TF lookup attempt {attempt+1}/20 for '{tf2Name}' failed: {e}")
                    if badPos is None:
                        self.get_logger().error(f"Could not resolve TF frame '{tf2Name}' after 20 attempts")
                        return
                    goodPos, goodEuler = self.to_good_frame(badPos, badEuler)

                    self.get_logger().info(f'Moving to marker ID {markerID} at pose: {bad_pos}')
                    self.get_logger().warn(
                        f"[moveToMarker] TF-based handle pos in base_link: {np.round(badPos, 4)} "
                        f"(marker base_link pos was {np.round(bad_pos, 4)}, "
                        f"expected handle ~= marker + offset {np.round(offsetPos, 4)})"
                    )
                    print(f"Computed target pose: pos={goodPos}, orient={goodEuler}")
                    self.freeze_markers()
                    self.move_to_pose(goodPos, goodEuler)
                    self.unfreeze_markers()
                    foundMarker = 1
                    return

            if not foundMarker:
                self.get_logger().warn(f"Marker ID {markerID} not found in detected marker poses.")
                available_ids = [entry['id'] for entry in markers]
                self.get_logger().info(f"Available marker IDs: {available_ids}")
        else:
            self.get_logger().warn("No marker poses available to move to marker.")

    def liftPlate(self, markerID=0):
        offsetPos = self.markerToPickupOffset
        offsetOri = self.offsetOri
        markers = self.marker_poses
        if markers:
            for entry in markers:
                if entry['id'] == markerID:
                    bad_pos = entry['positionInBase']
                    bad_euler = entry['eulerInBase']
                    tf2Name = entry['tf2Name']

                    # Re-broadcast marker TF and poll until the frame is available
                    badPos, badEuler = None, None
                    for attempt in range(20):
                        self.broadcast_marker_transform(bad_pos, bad_euler, child_frame=tf2Name)
                        time.sleep(0.05)
                        try:
                            badPos, badEuler = self.applyFrameChange(offsetPos, offsetOri, source_frame="base_link", target_frame=tf2Name)
                            break
                        except Exception as e:
                            self.get_logger().warn(f"TF lookup attempt {attempt+1}/20 for '{tf2Name}' failed: {e}")
                    if badPos is None:
                        self.get_logger().error(f"Could not resolve TF frame '{tf2Name}' after 20 attempts")
                        return
                    goodPos, goodEuler = self.to_good_frame(badPos, badEuler)

                    self.get_logger().info(f'Moving to marker ID {markerID} at pose: {bad_pos}')
                    self.freeze_markers()
                    self.move_to_pose(goodPos, goodEuler)
                    self.unfreeze_markers()
                    return

    def add_movement_constraint(self, constraint_type, value):
        """
        Add a path constraint without moving.
        constraint_type: 'orientation' or 'x', 'y', 'z'
        value: for 'orientation', tuple (x,y,z,w) quaternion
               for 'x','y','z', float value to constrain to
        """
        if constraint_type == 'orientation':
            # Assume value is (x,y,z,w) quaternion tuple
            self.poseReader.moveit2.set_path_orientation_constraint(
                quat_xyzw=value,
                frame_id="base_link",
                target_link=self.poseReader.end_effector_name,
                tolerance=0.15
            )
        elif constraint_type in ['x', 'y', 'z']:
            # Get current position
            self.poseReader.get_fk()
            current_pos = self.poseReader.pose[:3]  # [x, y, z]
            axis_index = {'x': 0, 'y': 1, 'z': 2}[constraint_type]
            
            # Set constrained position
            constrained_pos = current_pos.copy()
            constrained_pos[axis_index] = value
            
            # Create position constraint with box for anisotropic tolerance
            constraint = PositionConstraint()
            constraint.header.frame_id = "base_link"
            constraint.link_name = self.poseReader.end_effector_name
            constraint.constraint_region = BoundingVolume()
            constraint.constraint_region.primitives.append(SolidPrimitive())
            constraint.constraint_region.primitives[0].type = 1  # BOX
            # Dimensions: half-sizes
            tol_small = 0.001  # tight constraint on the axis
            tol_large = 10.0   # loose on others
            dimensions = [tol_large, tol_large, tol_large]
            dimensions[axis_index] = tol_small
            constraint.constraint_region.primitives[0].dimensions = dimensions
            constraint.constraint_region.primitive_poses.append(Pose())
            constraint.constraint_region.primitive_poses[0].position.x = constrained_pos[0]
            constraint.constraint_region.primitive_poses[0].position.y = constrained_pos[1]
            constraint.constraint_region.primitive_poses[0].position.z = constrained_pos[2]
            constraint.weight = 1.0
            
            # Append to path constraints
            self.poseReader.moveit2._MoveIt2__move_action_goal.request.path_constraints.position_constraints.append(constraint)
        else:
            self.get_logger().error(f"Unknown constraint type: {constraint_type}")


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
    print("  6) Add movement constraint")
    print("  7) Print current end-effector pose")
    print("  8) Set marker offsets")
    print("  9) Toggle marker updates")
    print(" 10) Scan to marker (by ID, uses TF)")
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
            markers = node.marker_poses  # <-- reads from stream.found_markers
            if markers:
                print(f"\n  Found {len(markers)} marker(s):")
                for entry in markers:
                    pos = entry.get('positionInWorld', 'N/A')
                    ori = entry.get('orientInWorld', 'N/A')
                    print(f"    ID {entry['id']}: pos={pos}, orient={ori}")
            else:
                print("  No markers found yet.")

        elif choice == "6":
            ctype = input("  Constraint type (orientation / x / y / z): ").strip()
            if ctype == "orientation":
                val = _parse_floats("  Quaternion (x y z w): ", 4)
                if val:
                    node.add_movement_constraint('orientation', tuple(val))
            elif ctype in ['x', 'y', 'z']:
                val = _parse_floats(f"  {ctype} value: ", 1)
                if val:
                    node.add_movement_constraint(ctype, val[0])
            else:
                print(f"  Unknown constraint type: {ctype}")

        elif choice == "7":
            if hasattr(node, 'pose') and node.pose is not None:
                print(f"  Current EEF pose: {node.pose}")
            else:
                print("  Pose not yet available (waiting for joint_states).")

        elif choice == "8":
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

        elif choice == "9":
            if node.stream.marker_updates_enabled:
                node.freeze_markers()
                print("  Marker updates FROZEN.")
            else:
                node.unfreeze_markers()
                print("  Marker updates RESUMED.")

        elif choice == "10":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            dist = _parse_floats("  Viewing distance [0.15]: ", 1)
            dist = dist[0] if dist else 0.15
            node.get_logger().info(f"User requested scanToMarker({mid}, dist={dist})")
            node.scanToMarker(marker_id=mid, viewing_distance=dist)

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
    runVirtual = 1

    if runVirtual:
        stream_source = "ros"  # Use ROS topic stream for simulated environment
        node = printerAutomation(calibration_mode=False,stream_source=stream_source)
        node.gripper_disabled = True  # Workaround: gripper causes joint drift in simulation

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
            pos=[0.37, -0.2, 0.21],
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