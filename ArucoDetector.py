#!/usr/bin/env python3

import rclpy
import numpy as np
import subprocess
import threading
import time as _time
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Point, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge
from rclpy.time import Time
import tf2_ros
from rclpy.callback_groups import ReentrantCallbackGroup
from tf2_geometry_msgs import do_transform_pose

from webVideoServer import WebVideoStream
from poseReader import PoseReader
from simulated3DPrinter import Simulated3DPrinter


class ArucoDetectionViewer(PoseReader):

    def __init__(self,
                 source="ros",
                 camera_index=None,
                 camera_keyword="GENERAL WEBCAM",
                 color_topic="/rgbd_camera/image",
                 depth_topic="/rgbd_camera/depth_image",
                 camera_info_topic="/rgbd_camera/camera_info",
                 feed_rotation_deg=0.0):
        super().__init__('aruco_detection_viewer', enable_pose_print=False)

        self.fps = 30.0
        self.markerNamePrefix = "aruco_marker_"
        self.filterStates = np.zeros((100, 6))

        # Default dt in case joint_states hasn't arrived yet
        if not hasattr(self, 'dt') or self.dt is None:
            self.dt = 1.0 / self.fps

        # Single object handles web server + aruco detection + frame compositing.
        # Pass through source / camera options so caller can choose "ros" or "webcam"
        self.stream = WebVideoStream(
            source=source,
            port=5000,
            fps=self.fps,
            display_scale=2.0,
            depth_colormap="turbo",
            marker_sizes=[0.03, 0.05],
            dict_names=['DICT_4X4_50', 'DICT_6X6_50'],
            enrich_fn=self._enrich_marker_pose,
            log_fn=lambda msg: self.get_logger().info(msg),
            camera_index=camera_index,
            camera_keyword=camera_keyword,
            color_topic=color_topic,
            depth_topic=depth_topic,
            camera_info_topic=camera_info_topic,
            feed_rotation_deg=feed_rotation_deg,
        )

        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer, self)

        # RViz visualization publisher
        self._marker_array_pub = self.create_publisher(MarkerArray, '/aruco_markers_viz', 10)
        self._rviz_publish_thread = threading.Thread(target=self._rviz_publish_loop, daemon=True)
        self._rviz_publish_thread.start()

        # Togglable 1-second timer for printing marker_poses
        self._marker_print_timer = self.create_timer(1.0, self._print_marker_poses)
        self._marker_print_enabled = False
        self._marker_print_timer.cancel()

    # ---- Marker enrichment ----

    def _enrich_marker_pose(self, entry: dict) -> dict:
        badPos, badEuler = self.cameraToBase(entry['positionFromCamera'], entry['eulerFromCamera'],
                                              markerID=entry['id'])
        if badPos is None:
            return None
        entry['tf2Name'] = f"{self.markerNamePrefix}{entry['id']}"

        # Store base_link frame values (from TF2) for RViz and TF operations
        entry['positionInBase'] = badPos
        entry['eulerInBase'] = badEuler

        # Compute user-facing "world" values using frame rotation only (no EEF offset angles)
        R_BF_GF = R.from_euler("XYZ", self.frameRotationAngles, degrees=False)
        goodPos = R_BF_GF.apply(badPos)
        goodEuler = (R_BF_GF * R.from_euler("XYZ", badEuler, degrees=False)).as_euler("XYZ", degrees=False)
        entry['positionInWorld'] = goodPos
        entry['orientInWorld'] = {'roll': np.degrees(goodEuler[0]), 'pitch': np.degrees(goodEuler[1]), 'yaw': np.degrees(goodEuler[2])}
        return entry

    # ---- Frame transforms ----

    def applyFrameChange(self, posInFrame, eulerInFrame,
                         source_frame="base_link", target_frame="ee_camera_link"):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(posInFrame[0]), float(posInFrame[1]), float(posInFrame[2])
        q = R.from_euler("XYZ", eulerInFrame, degrees=False).as_quat()
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = float(q[0]), float(q[1]), float(q[2]), float(q[3])

        transform = self.tf2_buffer.lookup_transform(source_frame, target_frame, Time())
        transformed = do_transform_pose(pose, transform)

        tf2_quat = [transformed.orientation.x, transformed.orientation.y,
                     transformed.orientation.z, transformed.orientation.w]
        return (np.array([transformed.position.x, transformed.position.y, transformed.position.z]),
                R.from_quat(tf2_quat).as_euler("XYZ", degrees=False))

    def cameraToBase(self, posInFrame, eulerInFrame, markerID=0):
        # Guard: if dt is not yet set by PoseReader, use default
        if not hasattr(self, 'dt') or self.dt is None or self.dt == 0:
            self.dt = 1.0 / self.fps

        try:
            badPos, badEuler = self.applyFrameChange(posInFrame, eulerInFrame,
                                                     source_frame="base_link", target_frame="ee_camera_link")
        except Exception:
            return None, None

        # Low-pass filter
        fCutoff = 3.0
        RC = 1 / (2 * np.pi * fCutoff)
        alpha = self.dt / (RC + self.dt)
        prev = self.filterStates[markerID, :]
        # Initialize filter to first measurement instead of zero
        if np.allclose(prev, 0.0):
            filteredPos = badPos
            filteredEuler = badEuler
        else:
            filteredPos = alpha * badPos + (1 - alpha) * prev[0:3]
            filteredEuler = alpha * badEuler + (1 - alpha) * prev[3:6]
        self.filterStates[markerID, :] = np.hstack((filteredPos, filteredEuler))

        try:
            self.broadcast_marker_transform(filteredPos, filteredEuler,
                                            child_frame=f"{self.markerNamePrefix}{markerID}")
        except Exception:
            pass  # TF broadcast failure is non-fatal
        return filteredPos, filteredEuler

    def broadcast_marker_transform(self, marker_pos, marker_orient,
                                   parent_frame="base_link", child_frame="aruco_marker"):
        if not hasattr(self, 'tf2_broadcaster'):
            self.tf2_broadcaster = tf2_ros.TransformBroadcaster(self)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = float(marker_pos[0]), float(marker_pos[1]), float(marker_pos[2])
        q = R.from_euler("XYZ", marker_orient, degrees=False).as_quat()
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        self.tf2_broadcaster.sendTransform(t)

    # ---- RViz marker visualization ----

    def _rviz_publish_loop(self):
        """Background thread that publishes RViz markers at ~10 Hz."""
        while rclpy.ok():
            try:
                self._publish_rviz_markers_impl()
            except Exception as e:
                self.get_logger().warn(f'RViz marker publish failed: {e}', throttle_duration_sec=5.0)
            _time.sleep(0.1)

    def _publish_rviz_markers_impl(self):
        """Inner implementation — separated so exceptions are caught by the wrapper."""
        msg = MarkerArray()
        now = self.get_clock().now().to_msg()
        next_id = 0  # global unique id across all marker sub-parts
        frame_id = 'base_link'  # markers are in base_link frame; RViz resolves to display frame via TF

        for entry in self.stream.marker_poses:
            if 'positionInBase' not in entry or 'eulerInBase' not in entry:
                continue

            # Use base_link values directly — RViz uses TF to transform to display frame
            pos = entry['positionInBase']
            euler_rad = entry['eulerInBase']
            q = R.from_euler('XYZ', euler_rad).as_quat()
            rot = R.from_euler('XYZ', euler_rad)
            size = entry.get('marker_size', 0.05)
            marker_id = int(entry['id'])
            dict_name = entry.get('dict_name', 'unknown')
            # Extract short type like "4x4" from "DICT_4X4_50"
            dict_short = dict_name.replace('DICT_', '').rsplit('_', 1)[0].replace('X', 'x')

            # --- Flat rectangular prism ---
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = frame_id
            m.ns = 'aruco_body'
            m.id = next_id; next_id += 1
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            m.pose.orientation.x = float(q[0])
            m.pose.orientation.y = float(q[1])
            m.pose.orientation.z = float(q[2])
            m.pose.orientation.w = float(q[3])
            m.scale.x = float(size)
            m.scale.y = float(size)
            m.scale.z = 0.002
            m.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            msg.markers.append(m)

            thickness = 0.002
            # Local Z axis of the marker = normal to front face
            normal = rot.apply([0.0, 0.0, 1.0])
            front_offset = normal * (thickness / 2.0 + 0.001)
            back_offset = -front_offset

            # --- "front" text on front face ---
            ft = Marker()
            ft.header.stamp = now
            ft.header.frame_id = frame_id
            ft.ns = 'aruco_front'
            ft.id = next_id; next_id += 1
            ft.type = Marker.TEXT_VIEW_FACING
            ft.action = Marker.ADD
            ft.pose.position.x = float(pos[0] + front_offset[0])
            ft.pose.position.y = float(pos[1] + front_offset[1])
            ft.pose.position.z = float(pos[2] + front_offset[2])
            ft.scale.z = float(size * 0.3)
            ft.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
            ft.text = 'front'
            msg.markers.append(ft)

            # --- "back" text on back face ---
            bt = Marker()
            bt.header.stamp = now
            bt.header.frame_id = frame_id
            bt.ns = 'aruco_back'
            bt.id = next_id; next_id += 1
            bt.type = Marker.TEXT_VIEW_FACING
            bt.action = Marker.ADD
            bt.pose.position.x = float(pos[0] + back_offset[0])
            bt.pose.position.y = float(pos[1] + back_offset[1])
            bt.pose.position.z = float(pos[2] + back_offset[2])
            bt.scale.z = float(size * 0.3)
            bt.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)
            bt.text = 'back'
            msg.markers.append(bt)

            # --- ID + type label above marker ---
            lt = Marker()
            lt.header.stamp = now
            lt.header.frame_id = frame_id
            lt.ns = 'aruco_labels'
            lt.id = next_id; next_id += 1
            lt.type = Marker.TEXT_VIEW_FACING
            lt.action = Marker.ADD
            lt.pose.position.x = float(pos[0])
            lt.pose.position.y = float(pos[1])
            lt.pose.position.z = float(pos[2]) + float(size) + 0.02
            lt.scale.z = 0.03
            lt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            lt.text = f'ID:{marker_id} ({dict_short})'
            msg.markers.append(lt)

            # --- Debug: show raw coordinates used for this marker ---
            euler_deg = np.degrees(euler_rad)
            debug_lines = [
                f'p=[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]',
                f'[{euler_deg[0]:.0f},{euler_deg[1]:.0f},{euler_deg[2]:.0f}]',
                f'q=[{q[0]:.2f},{q[1]:.2f},{q[2]:.2f},{q[3]:.2f}]',
            ]
            line_spacing = 0.018
            base_z = float(pos[2]) - float(size) - 0.01
            for i, text_line in enumerate(debug_lines):
                dbg = Marker()
                dbg.header.stamp = now
                dbg.header.frame_id = frame_id
                dbg.ns = 'aruco_debug'
                dbg.id = next_id; next_id += 1
                dbg.type = Marker.TEXT_VIEW_FACING
                dbg.action = Marker.ADD
                dbg.pose.position.x = float(pos[0])
                dbg.pose.position.y = float(pos[1])
                dbg.pose.position.z = base_z - i * line_spacing
                dbg.scale.z = 0.01
                dbg.color = ColorRGBA(r=0.8, g=0.8, b=1.0, a=1.0)
                dbg.text = text_line
                msg.markers.append(dbg)

            # --- Coordinate axes (X=red, Y=green, Z=blue) ---
            axis_len = float(size * 0.8)
            axis_radius = 0.003
            axes = [
                ([1, 0, 0], ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)),  # X
                ([0, 1, 0], ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)),  # Y
                ([0, 0, 1], ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)),  # Z
            ]
            for local_dir, color in axes:
                world_dir = rot.apply(local_dir)
                # Arrow start = marker center, end = center + axis_len * direction
                ax = Marker()
                ax.header.stamp = now
                ax.header.frame_id = frame_id
                ax.ns = 'aruco_axes'
                ax.id = next_id; next_id += 1
                ax.type = Marker.ARROW
                ax.action = Marker.ADD
                # Use start/end points
                start = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
                end = Point(
                    x=float(pos[0] + world_dir[0] * axis_len),
                    y=float(pos[1] + world_dir[1] * axis_len),
                    z=float(pos[2] + world_dir[2] * axis_len),
                )
                ax.points = [start, end]
                ax.scale.x = axis_radius * 2   # shaft diameter
                ax.scale.y = axis_radius * 4   # head diameter
                ax.scale.z = 0.0               # auto head length
                ax.color = color
                msg.markers.append(ax)

        if msg.markers:
            self._marker_array_pub.publish(msg)
        else:
            self.get_logger().warn('RViz: no markers with positionInBase to publish', throttle_duration_sec=5.0)

    # ---- Marker pose printing ----

    def enable_marker_print(self):
        """Start printing marker_poses to console every second."""
        if not self._marker_print_enabled:
            self._marker_print_enabled = True
            self._marker_print_timer.reset()
            self.get_logger().info('Marker pose printing enabled')

    def disable_marker_print(self):
        """Stop printing marker_poses to console."""
        if self._marker_print_enabled:
            self._marker_print_enabled = False
            self._marker_print_timer.cancel()
            self.get_logger().info('Marker pose printing disabled')

    def toggle_marker_print(self):
        """Toggle marker_poses console printing on/off."""
        if self._marker_print_enabled:
            self.disable_marker_print()
        else:
            self.enable_marker_print()

    def _print_marker_poses(self):
        poses = self.marker_poses
        if not poses:
            self.get_logger().info('[marker_poses] No markers seen yet')
            return
        for m in poses:
            gp = m.get('global_pose')
            if gp:
                pos = gp['position']
                ori = gp['orientation']
                self.get_logger().info(
                    f"[marker_poses] ID:{m['id']}  dict:{m['dict_name']}  "
                    f"pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  "
                    f"rpy=({ori['roll']:+.1f}, {ori['pitch']:+.1f}, {ori['yaw']:+.1f})")
            else:
                self.get_logger().info(
                    f"[marker_poses] ID:{m['id']}  dict:{m['dict_name']}  global_pose: N/A")

    @property
    def marker_poses(self):
        """Persistent list of all markers ever detected, with id, dict size, and global pose."""
        result = []
        for entry in self.stream.marker_poses:
            item = dict(entry)
            item['dict_name'] = entry.get('dict_name', 'unknown')
            if 'positionInWorld' in entry and 'orientInWorld' in entry:
                item['global_pose'] = {
                    'position': entry['positionInWorld'],
                    'orientation': entry['orientInWorld'],
                }
            result.append(item)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectionViewer()
    node.enable_marker_print()
    printer = Simulated3DPrinter(
        node=node,
        pos=[0.0, -0.67, 0.38],
        orient=[0.0, 0.0, np.pi],
    )
    printer.spawn_fast()
    
    

    node.move_to_pose(np.array([0.4,0.0,0.4]), np.array([0.0,0.0,0.0]))
    # Spin both the ROS node (PoseReader/tf2) and the stream's internal ROS node
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(node.stream._ros_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        node.stream._ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()