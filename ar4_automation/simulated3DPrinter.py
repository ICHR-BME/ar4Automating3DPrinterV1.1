import os
import atexit
import subprocess
import tempfile
import time
import math
import threading
import re
import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, PointStamped, PoseStamped, Pose, Point, Quaternion
import tf_transformations
import tf2_geometry_msgs
import tf2_ros
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity





class Simulated3DPrinter:
    """Simulated 3D printer in Gazebo: box of walls, swinging door, ArUco markers.

    door_marker_local_pos is an offset from the door front face center (-y is
    outward); static_marker_local_pos is in the printer frame (None = default).
    """
    
    # models/ lives at the repo root, one level above this package
    DEFAULT_MODEL_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models', 'aruco_marker', '')
    
    def __init__(
        self,
        node=None,
        pos=(0.0, -0.63, 0.15)+np.random.uniform(-0.03, 0.03, size=3),
        orient=(0.0, 0.0, np.pi),
        width=0.2,
        depth=0.2,
        height=0.3,
        wall_thickness=0.01,
        door_frequency=0.2,
        door_amplitude=math.pi / 2,
        door_marker_texture='materials/textures/marker6x6_0.png',
        door_marker_size=0.05,
        door_marker_local_pos=[0,0,0.01],
        static_marker_texture='materials/textures/marker4x4_0.png',
        static_marker_size=0.03,
        static_marker_local_pos=None,
        enable_door_flapping_animation=False,
        model_dir=None,
        use_bad_frame=True
    ):
        self._owns_node = node is None
        self.node = node if node else Node('simulated_printer')
        
        pos = np.array(pos)
        orient = np.array(orient)

        # good frame -> bad frame (base_link/world). Only the frame rotation,
        # not the EEF offset angles that to_bad_frame adds.
        if use_bad_frame and hasattr(self.node, 'frameRotationAngles'):
            from scipy.spatial.transform import Rotation as R_scipy
            R_BF_GF = R_scipy.from_euler("XYZ", self.node.frameRotationAngles, degrees=False)
            R_GF_BF = R_BF_GF.inv()
            pos = R_GF_BF.apply(pos)
            R_orient = R_scipy.from_euler("XYZ", orient, degrees=False)
            orient = (R_GF_BF * R_orient).as_euler("XYZ", degrees=False)

        self.pos = np.array(pos)
        self.orient = np.array(orient)
        self.width = width
        self.depth = depth
        self.height = height
        self.wall_thickness = wall_thickness
        self.door_frequency = door_frequency
        self.door_amplitude = door_amplitude
        self.model_dir = model_dir or self.DEFAULT_MODEL_DIR
        
        self.door_marker_texture = door_marker_texture
        self.door_marker_size = door_marker_size
        self.door_marker_local_pos = door_marker_local_pos
        self.static_marker_texture = static_marker_texture
        self.static_marker_size = static_marker_size
        self.static_marker_local_pos = static_marker_local_pos
        self.enable_door_flapping_animation = enable_door_flapping_animation

        # instance ID from the trailing number in the marker texture filename
        # ("marker6x6_0.png" -> "0"), full stem if there are no digits
        _stem = os.path.basename(door_marker_texture).rsplit('.', 1)[0]
        _match = re.search(r'(\d+)$', _stem)
        self._instance_id = _match.group(1) if _match else _stem
        
        self.tf_broadcaster = TransformBroadcaster(self.node)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)

        self.set_pose_client = None
        self._bridge_proc = None

        self.q = tf_transformations.quaternion_from_euler(
            float(self.orient[0]), float(self.orient[1]), float(self.orient[2])
        )
        self.q = [float(x) for x in self.q]
        self.q_marker = tf_transformations.quaternion_multiply(
            self.q, tf_transformations.quaternion_from_euler(0, 0, math.pi/2)
        )
        self.q_marker = [float(x) for x in self.q_marker]
        
        # spawned entity names, kept for cleanup
        self.spawned_entities = []
        self.markers = {}
        self.walls = {}

        self.door_pose_pub = None
        self.marker_pose_pub = None

        self.animation_thread = None
        self.running = False

        # set during spawn
        self._door_name = None
        self._door_marker_name = None

    def rotate_vector_by_quaternion(self, v, q):
        """Rotate a 3D vector by a quaternion."""
        rot_matrix = tf_transformations.quaternion_matrix(q)[:3, :3]
        return rot_matrix @ v
    
    def _setup_pose_service(self):
        """Initialize the ros_gz_bridge and service client.

        Reuse an already-running bridge when possible: every leaked
        parameter_bridge subprocess registers as the SAME node name
        (/ros_gz_bridge), and duplicate node names corrupt DDS discovery —
        symptom: fresh nodes whose camera_info subscription silently never
        matches, so aruco detection runs uncalibrated and spots nothing."""
        self._bridge_proc = None
        self.set_pose_client = self.node.create_client(SetEntityPose, '/world/default/set_pose')
        if self.set_pose_client.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().info('SetEntityPose service already available (reusing bridge)')
            return

        self._bridge_proc = subprocess.Popen(
            ['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
             '/world/default/set_pose@ros_gz_interfaces/srv/SetEntityPose',
             '--ros-args', '-r', f'__node:=set_pose_bridge_{os.getpid()}'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        atexit.register(self._bridge_proc.terminate)
        self.node.get_logger().info('Started ros_gz_bridge for set_pose service')
        if self.set_pose_client.wait_for_service(timeout_sec=10.0):
            self.node.get_logger().info('SetEntityPose service available')
        else:
            self.node.get_logger().warn('SetEntityPose service not available')

    def _broadcast_printer_frame(self):
        """Broadcast the printer's TF frame."""
        transform = TransformStamped()
        transform.header.stamp = self.node.get_clock().now().to_msg()
        transform.header.frame_id = "world"
        transform.child_frame_id = f"printer_frame_{self._instance_id}"
        transform.transform.translation.x = float(self.pos[0])
        transform.transform.translation.y = float(self.pos[1])
        transform.transform.translation.z = float(self.pos[2])
        transform.transform.rotation.x = self.q[0]
        transform.transform.rotation.y = self.q[1]
        transform.transform.rotation.z = self.q[2]
        transform.transform.rotation.w = self.q[3]
        self.tf_broadcaster.sendTransform(transform)
        return transform

    def _spin_or_sleep(self, timeout_sec):
        """Process callbacks for ~timeout_sec. Never spin_once() a node a
        background executor owns (it detaches the node and kills callback
        delivery for good); just sleep in that case."""
        if self.node.executor is None:
            rclpy.spin_once(self.node, timeout_sec=timeout_sec)
        else:
            time.sleep(timeout_sec)

    def _get_transform_with_retry(self, target_frame, source_frame, timeout_sec=5.0):
        """Get a TF transform with retry logic."""
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            try:
                return self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self._broadcast_printer_frame()
                self._spin_or_sleep(0.1)
        raise tf2_ros.LookupException(
            f"Could not get transform from {source_frame} to {target_frame} after {timeout_sec}s"
        )

    def _transform_point_to_world(self, local_x, local_y, local_z):
        """Transform a point from printer frame to world frame."""
        point_stamped = PointStamped()
        point_stamped.header.frame_id = f"printer_frame_{self._instance_id}"
        point_stamped.header.stamp = self.node.get_clock().now().to_msg()
        point_stamped.point.x = float(local_x)
        point_stamped.point.y = float(local_y)
        point_stamped.point.z = float(local_z)

        transform = self._get_transform_with_retry("world", f"printer_frame_{self._instance_id}")
        transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
        return transformed.point.x, transformed.point.y, transformed.point.z

    def _delete_entity(self, name):
        """Delete an entity from Gazebo."""
        delete_cmd = [
            'gz', 'service', '-s', '/world/default/remove',
            '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000', '--req', f'name: "{name}" type: MODEL'
        ]
        try:
            result = subprocess.run(delete_cmd, capture_output=True, text=True, check=True)
            msg = result.stdout.strip() or f'{name} deleted successfully'
            self.node.get_logger().debug(f'Delete: {msg}')
        except subprocess.CalledProcessError as e:
            pass  # entity may not exist
        except Exception as e:
            self.node.get_logger().warn(f'Delete exception for {name}: {e}')

    def _generate_combined_sdf(self, model_name, include_door=True):
        """Build one SDF model with all walls and markers as visuals in a single link."""
        t = self.wall_thickness
        w, d, h = self.width, self.depth, self.height

        # (name, size, local_pos) in the printer's local frame
        wall_configs = [
            ("bottom", [w, d, t], [0, 0, -h/2]),
            ("top", [w, d, t], [0, 0, h/2]),
            ("left", [t, d, h], [-w/2, 0, 0]),
            ("right", [t, d, h], [w/2, 0, 0]),
            ("back", [w, t, h], [0, d/2, 0]),
        ]
        if include_door:
            wall_configs.append(("front", [w, t, h], [0, -d/2, 0]))
        
        visuals = ""
        for name, size, local_pos in wall_configs:
            visuals += f"""
      <visual name="{name}">
        <pose>{local_pos[0]} {local_pos[1]} {local_pos[2]} 0 0 0</pose>
        <geometry>
          <box>
            <size>{size[0]} {size[1]} {size[2]}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
        </material>
      </visual>"""
        
        if self.static_marker_texture:
            static_local_pos = self.static_marker_local_pos
            if static_local_pos is None:
                static_local_pos = [0, -d/2 - t + 0.05, 0]
            # marker needs 90 deg yaw
            visuals += f"""
      <visual name="static_marker">
        <pose>{static_local_pos[0]} {static_local_pos[1]} {static_local_pos[2]} 0 0 {math.pi/2}</pose>
        <geometry>
          <box>
            <size>0.0001 {self.static_marker_size} {self.static_marker_size}</size>
          </box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <pbr>
            <metal>
              <albedo_map>{self.static_marker_texture}</albedo_map>
              <metalness>0.0</metalness>
              <roughness>1.0</roughness>
            </metal>
          </pbr>
        </material>
      </visual>"""
        
        if include_door and self.door_marker_texture:
            door_relative_pos = self.door_marker_local_pos if self.door_marker_local_pos else [0, 0, 0]
            marker_surface_offset = -0.005  # 5mm outside
            door_front_face_local = [0, -d/2 - t/2 + marker_surface_offset, 0]
            door_marker_pos = [
                door_front_face_local[0] + door_relative_pos[0],
                door_front_face_local[1] + door_relative_pos[1],
                door_front_face_local[2] + door_relative_pos[2]
            ]
            visuals += f"""
      <visual name="door_marker">
        <pose>{door_marker_pos[0]} {door_marker_pos[1]} {door_marker_pos[2]} 0 0 {math.pi/2}</pose>
        <geometry>
          <box>
            <size>0.0001 {self.door_marker_size} {self.door_marker_size}</size>
          </box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <pbr>
            <metal>
              <albedo_map>{self.door_marker_texture}</albedo_map>
              <metalness>0.0</metalness>
              <roughness>1.0</roughness>
            </metal>
          </pbr>
        </material>
      </visual>"""
        
        sdf_content = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="body">
{visuals}
    </link>
  </model>
</sdf>"""
        return sdf_content

    

    def _generate_door_sdf(self, door_name):
        """Generate SDF for the door (front wall) as a separate model."""
        t = self.wall_thickness
        w, h = self.width, self.height
        
        return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{door_name}">
    <static>false</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <box>
            <size>{w} {t} {h}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

    def _generate_door_marker_sdf(self, marker_name):
        """Generate SDF for the door marker as a separate model."""
        # relative path works because the SDF file is written into model_dir
        texture_path = self.door_marker_texture
        
        return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{marker_name}">
    <static>false</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <box>
            <size>0.0001 {self.door_marker_size} {self.door_marker_size}</size>
          </box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <pbr>
            <metal>
              <albedo_map>{texture_path}</albedo_map>
              <metalness>0.0</metalness>
              <roughness>1.0</roughness>
            </metal>
          </pbr>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

    def spawn_fast(self, body_name=None):
        """Spawn static parts as one model, door + marker separate for animation.

        3 spawn commands instead of 8+, much more reliable than spawn_complete().
        """
        if body_name is None:
            body_name = f"printer_body_{self._instance_id}"
        door_name = f"door_{self._instance_id}"

        self._setup_pose_service()

        # broadcast a few times so TF has the printer frame before we look it up
        for _ in range(10):
            self._broadcast_printer_frame()
            self._spin_or_sleep(0.1)

        # clear leftovers from either spawn variant
        self._delete_entity(body_name)
        self._delete_entity(door_name)
        for wall_name in ["bottom", "top", "left", "right", "front", "back"]:
            self._delete_entity(f"{wall_name}_{self._instance_id}")
        
        # 1. static body (5 walls + static marker)
        sdf_content = self._generate_combined_sdf(body_name, include_door=False)
        
        with tempfile.NamedTemporaryFile(dir=self.model_dir, mode='w', suffix='.sdf', delete=False) as f:
            f.write(sdf_content)
            body_sdf_path = f.name
        
        x, y, z = float(self.pos[0]), float(self.pos[1]), float(self.pos[2])
        roll, pitch, yaw = float(self.orient[0]), float(self.orient[1]), float(self.orient[2])
        
        cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-file', body_sdf_path, '-name', body_name,
            '-x', str(x), '-y', str(y), '-z', str(z),
            '-R', str(roll), '-P', str(pitch), '-Y', str(yaw)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.node.get_logger().info(f'Spawned printer body: {body_name}')
            self.spawned_entities.append(body_name)
        except subprocess.CalledProcessError as e:
            self.node.get_logger().error(f'Failed to spawn body: {e.stderr or e.stdout}')
        finally:
            os.unlink(body_sdf_path)
        
        # 2. door
        roll, pitch, yaw = float(self.orient[0]), float(self.orient[1]), float(self.orient[2])

        door_sdf = self._generate_door_sdf(door_name)
        with tempfile.NamedTemporaryFile(dir=self.model_dir, mode='w', suffix='.sdf', delete=False) as f:
            f.write(door_sdf)
            door_sdf_path = f.name

        door_local = [0, -self.depth/2, 0]
        door_x, door_y, door_z = self._transform_point_to_world(*door_local)

        cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-file', door_sdf_path, '-name', door_name,
            '-x', str(door_x), '-y', str(door_y), '-z', str(door_z),
            '-R', str(roll), '-P', str(pitch), '-Y', str(yaw)
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.node.get_logger().info(f'Spawned door: {door_name}')
            self.spawned_entities.append(door_name)
            self._door_name = door_name
        except subprocess.CalledProcessError as e:
            self.node.get_logger().error(f'Failed to spawn door: {e.stderr or e.stdout}')
        finally:
            os.unlink(door_sdf_path)
        
        # 3. door marker
        marker_name = os.path.basename(self.door_marker_texture).split('.')[0]
        self._delete_entity(marker_name)
        
        marker_sdf = self._generate_door_marker_sdf(marker_name)
        with tempfile.NamedTemporaryFile(dir=self.model_dir, mode='w', suffix='.sdf', delete=False) as f:
            f.write(marker_sdf)
            marker_sdf_path = f.name
        
        door_relative_pos = self.door_marker_local_pos if self.door_marker_local_pos else [0, 0, 0]
        marker_surface_offset = -0.005
        door_front_face_local = [0, -self.depth/2 - self.wall_thickness/2 + marker_surface_offset, 0]
        marker_local = [
            door_front_face_local[0] + door_relative_pos[0],
            door_front_face_local[1] + door_relative_pos[1],
            door_front_face_local[2] + door_relative_pos[2]
        ]
        marker_x, marker_y, marker_z = self._transform_point_to_world(*marker_local)
        marker_yaw = yaw + math.pi/2
        
        cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-file', marker_sdf_path, '-name', marker_name,
            '-x', str(marker_x), '-y', str(marker_y), '-z', str(marker_z),
            '-R', str(roll), '-P', str(pitch), '-Y', str(marker_yaw)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.node.get_logger().info(f'Spawned door marker: {marker_name}')
            self.spawned_entities.append(marker_name)
        except subprocess.CalledProcessError as e:
            self.node.get_logger().error(f'Failed to spawn marker: {e.stderr or e.stdout}')
        finally:
            os.unlink(marker_sdf_path)
        
        self._door_marker_name = marker_name

        if self.enable_door_flapping_animation:
            self.start_door_flapping_animation(marker_name)
        
        self.node.get_logger().info('Fast printer spawn complete (3 models)')
        return body_name, door_name, marker_name

    def get_door_marker_pose_in_base(self):
        """Door marker pose in base_link, ArUco convention (Z out of the
        face). Returns (bad_pos, bad_euler) matching ArucoDetector's frames."""
        from scipy.spatial.transform import Rotation as R_scipy

        # Marker local position in the printer frame
        door_relative_pos = self.door_marker_local_pos if self.door_marker_local_pos else [0, 0, 0]
        marker_surface_offset = -0.005
        door_front_face_local = np.array([
            0,
            -self.depth / 2 - self.wall_thickness / 2 + marker_surface_offset,
            0,
        ])
        marker_local_pos = door_front_face_local + np.array(door_relative_pos)

        # Printer orientation in world (bad frame)
        R_printer = R_scipy.from_euler("XYZ", self.orient, degrees=False)

        # Marker world position
        marker_world_pos = np.array(self.pos) + R_printer.apply(marker_local_pos)

        # ArUco-convention rotation: door faces -Y locally, so marker Z (out
        # of face) = -Y, marker X = +X, marker Y = Z x X (down)
        marker_z_local = np.array([0.0, -1.0, 0.0])
        marker_x_local = np.array([1.0, 0.0, 0.0])
        marker_y_local = np.cross(marker_z_local, marker_x_local)

        R_marker_local = np.column_stack([marker_x_local, marker_y_local, marker_z_local])
        R_marker_world = R_printer.as_matrix() @ R_marker_local
        marker_euler_world = R_scipy.from_matrix(R_marker_world).as_euler("XYZ", degrees=False)

        # world == base_link (both are the Gazebo fixed frame)
        return marker_world_pos, marker_euler_world

    def _animate_door(self, door_name, marker_name, local_marker_pos):
        """Animation loop for the swinging door."""
        offset_to_center = np.array([self.width/2, 0, 0])  # from hinge to door center
        
        # Compute hinge position in world
        hinge_local = [-self.width/2, -self.depth/2, 0]
        hinge_world = list(self._transform_point_to_world(*hinge_local))
        
        # Broadcast hinge frame
        hinge_transform = TransformStamped()
        hinge_transform.header.frame_id = "world"
        hinge_transform.child_frame_id = "hinge_frame"
        hinge_transform.transform.translation.x = hinge_world[0]
        hinge_transform.transform.translation.y = hinge_world[1]
        hinge_transform.transform.translation.z = hinge_world[2]
        hinge_transform.transform.rotation.x = self.q[0]
        hinge_transform.transform.rotation.y = self.q[1]
        hinge_transform.transform.rotation.z = self.q[2]
        hinge_transform.transform.rotation.w = self.q[3]
        
        while self.running:
            t = time.time()
            # Oscillate between 0 and -amplitude
            delta_yaw = -self.door_amplitude * (math.sin(2 * math.pi * self.door_frequency * t) + 1) / 2
            
            # Compute rotated offset
            cos_y, sin_y = math.cos(delta_yaw), math.sin(delta_yaw)
            rotated_offset = np.array([
                cos_y * offset_to_center[0] - sin_y * offset_to_center[1],
                sin_y * offset_to_center[0] + cos_y * offset_to_center[1],
                offset_to_center[2]
            ])
            
            # Door position and orientation in world
            door_pos = np.array(hinge_world) + self.rotate_vector_by_quaternion(rotated_offset, self.q)
            q_delta = tf_transformations.quaternion_from_euler(0, 0, delta_yaw)
            door_ori = tf_transformations.quaternion_multiply(self.q, q_delta)
            door_ori = [float(x) for x in door_ori]
            
            # Marker position and orientation (with extra 90° rotation)
            marker_pos = door_pos + self.rotate_vector_by_quaternion(local_marker_pos, door_ori)
            q_marker_offset = tf_transformations.quaternion_from_euler(0, 0, math.pi/2)
            marker_ori = tf_transformations.quaternion_multiply(door_ori, q_marker_offset)
            marker_ori = [float(x) for x in marker_ori]
            
            # Update Gazebo poses via service
            if self.set_pose_client:
                try:
                    # Door pose
                    door_request = SetEntityPose.Request()
                    door_request.entity = Entity()
                    door_request.entity.name = door_name
                    door_request.entity.type = Entity.MODEL
                    door_request.pose = Pose()
                    door_request.pose.position = Point(x=float(door_pos[0]), y=float(door_pos[1]), z=float(door_pos[2]))
                    door_request.pose.orientation = Quaternion(x=float(door_ori[0]), y=float(door_ori[1]), z=float(door_ori[2]), w=float(door_ori[3]))
                    self.set_pose_client.call_async(door_request)
                    
                    # Marker pose
                    marker_request = SetEntityPose.Request()
                    marker_request.entity = Entity()
                    marker_request.entity.name = marker_name
                    marker_request.entity.type = Entity.MODEL
                    marker_request.pose = Pose()
                    marker_request.pose.position = Point(x=float(marker_pos[0]), y=float(marker_pos[1]), z=float(marker_pos[2]))
                    marker_request.pose.orientation = Quaternion(x=float(marker_ori[0]), y=float(marker_ori[1]), z=float(marker_ori[2]), w=float(marker_ori[3]))
                    self.set_pose_client.call_async(marker_request)
                except Exception as e:
                    self.node.get_logger().warn(f'Failed to set pose: {e}')
            
            # Publish poses for other nodes
            if self.door_pose_pub:
                door_msg = PoseStamped()
                door_msg.header.stamp = self.node.get_clock().now().to_msg()
                door_msg.header.frame_id = "world"
                door_msg.pose.position = Point(x=float(door_pos[0]), y=float(door_pos[1]), z=float(door_pos[2]))
                door_msg.pose.orientation = Quaternion(x=float(door_ori[0]), y=float(door_ori[1]), z=float(door_ori[2]), w=float(door_ori[3]))
                self.door_pose_pub.publish(door_msg)
            
            if self.marker_pose_pub:
                marker_msg = PoseStamped()
                marker_msg.header.stamp = self.node.get_clock().now().to_msg()
                marker_msg.header.frame_id = "world"
                marker_msg.pose.position = Point(x=float(marker_pos[0]), y=float(marker_pos[1]), z=float(marker_pos[2]))
                marker_msg.pose.orientation = Quaternion(x=float(marker_ori[0]), y=float(marker_ori[1]), z=float(marker_ori[2]), w=float(marker_ori[3]))
                self.marker_pose_pub.publish(marker_msg)
            
            # Broadcast TF frames
            hinge_transform.header.stamp = self.node.get_clock().now().to_msg()
            self.tf_broadcaster.sendTransform(hinge_transform)
            
            door_transform = TransformStamped()
            door_transform.header.stamp = self.node.get_clock().now().to_msg()
            door_transform.header.frame_id = "hinge_frame"
            door_transform.child_frame_id = "door_frame"
            door_transform.transform.translation.x = float(rotated_offset[0])
            door_transform.transform.translation.y = float(rotated_offset[1])
            door_transform.transform.translation.z = float(rotated_offset[2])
            door_transform.transform.rotation.x = float(q_delta[0])
            door_transform.transform.rotation.y = float(q_delta[1])
            door_transform.transform.rotation.z = float(q_delta[2])
            door_transform.transform.rotation.w = float(q_delta[3])
            self.tf_broadcaster.sendTransform(door_transform)
            
            marker_transform = TransformStamped()
            marker_transform.header.stamp = self.node.get_clock().now().to_msg()
            marker_transform.header.frame_id = "door_frame"
            marker_transform.child_frame_id = "marker_frame"
            marker_transform.transform.translation.x = float(local_marker_pos[0])
            marker_transform.transform.translation.y = float(local_marker_pos[1])
            marker_transform.transform.translation.z = float(local_marker_pos[2])
            marker_transform.transform.rotation.x = 0.0
            marker_transform.transform.rotation.y = 0.0
            marker_transform.transform.rotation.z = 0.0
            marker_transform.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(marker_transform)
            
            time.sleep(0.02)  # 50Hz

    def _set_door_angle(self, door_name, marker_name, angle):
        """Set the door to a specific angle (0 = closed, -amplitude = fully open)."""
        offset_to_center = np.array([self.width/2, 0, 0])
        
        # Marker offset from door center: front face offset + surface offset + user-specified offset
        # Front face is at y = -wall_thickness/2 from door center, plus 5mm outside
        marker_surface_offset = -0.005  # 5mm outside the door
        user_offset = np.array(self.door_marker_local_pos) if self.door_marker_local_pos is not None else np.array([0, 0, 0])
        local_marker_pos = np.array([0, -self.wall_thickness/2 + marker_surface_offset, 0]) + user_offset
        
        # Compute hinge position in world
        hinge_local = [-self.width/2, -self.depth/2, 0]
        hinge_world = list(self._transform_point_to_world(*hinge_local))
        
        # Compute rotated offset
        cos_y, sin_y = math.cos(angle), math.sin(angle)
        rotated_offset = np.array([
            cos_y * offset_to_center[0] - sin_y * offset_to_center[1],
            sin_y * offset_to_center[0] + cos_y * offset_to_center[1],
            offset_to_center[2]
        ])
        
        # Door position and orientation in world
        door_pos = np.array(hinge_world) + self.rotate_vector_by_quaternion(rotated_offset, self.q)
        q_delta = tf_transformations.quaternion_from_euler(0, 0, angle)
        door_ori = tf_transformations.quaternion_multiply(self.q, q_delta)
        door_ori = [float(x) for x in door_ori]
        
        # Marker position and orientation (with extra 90° rotation)
        marker_pos = door_pos + self.rotate_vector_by_quaternion(local_marker_pos, door_ori)
        q_marker_offset = tf_transformations.quaternion_from_euler(0, 0, math.pi/2)
        marker_ori = tf_transformations.quaternion_multiply(door_ori, q_marker_offset)
        marker_ori = [float(x) for x in marker_ori]
        
        # Update Gazebo poses via service
        if self.set_pose_client:
            try:
                # Door pose
                door_request = SetEntityPose.Request()
                door_request.entity = Entity()
                door_request.entity.name = door_name
                door_request.entity.type = Entity.MODEL
                door_request.pose = Pose()
                door_request.pose.position = Point(x=float(door_pos[0]), y=float(door_pos[1]), z=float(door_pos[2]))
                door_request.pose.orientation = Quaternion(x=float(door_ori[0]), y=float(door_ori[1]), z=float(door_ori[2]), w=float(door_ori[3]))
                self.set_pose_client.call_async(door_request)
                
                # Marker pose
                marker_request = SetEntityPose.Request()
                marker_request.entity = Entity()
                marker_request.entity.name = marker_name
                marker_request.entity.type = Entity.MODEL
                marker_request.pose = Pose()
                marker_request.pose.position = Point(x=float(marker_pos[0]), y=float(marker_pos[1]), z=float(marker_pos[2]))
                marker_request.pose.orientation = Quaternion(x=float(marker_ori[0]), y=float(marker_ori[1]), z=float(marker_ori[2]), w=float(marker_ori[3]))
                self.set_pose_client.call_async(marker_request)
            except Exception as e:
                self.node.get_logger().warn(f'Failed to set pose: {e}')

    def open_door(self, duration=1.0):
        """Animate the door to fully open over duration seconds."""
        if not hasattr(self, '_door_marker_name') or not self._door_marker_name:
            self.node.get_logger().warn('Door marker not spawned yet')
            return
        
        if not hasattr(self, '_current_door_angle'):
            self._current_door_angle = 0.0
        
        start_angle = self._current_door_angle
        end_angle = -self.door_amplitude
        self._animate_door_to_angle(start_angle, end_angle, duration)
        self._current_door_angle = end_angle
        self.node.get_logger().info('Door opened')

    def close_door(self, duration=1.0):
        """Animate the door closed over duration seconds."""
        if not hasattr(self, '_door_marker_name') or not self._door_marker_name:
            self.node.get_logger().warn('Door marker not spawned yet')
            return
        
        if not hasattr(self, '_current_door_angle'):
            self._current_door_angle = -self.door_amplitude
        
        start_angle = self._current_door_angle
        end_angle = 0.0
        self._animate_door_to_angle(start_angle, end_angle, duration)
        self._current_door_angle = end_angle
        self.node.get_logger().info('Door closed')

    def _animate_door_to_angle(self, start_angle, end_angle, duration, fps=50):
        """Animate the door from start_angle to end_angle (rad) over
        duration seconds at fps."""
        num_steps = int(duration * fps)
        if num_steps < 1:
            num_steps = 1
        
        # Use stored door name, default to "front" for spawn_complete() compatibility
        door_name = self._door_name if self._door_name else "front"
        
        for i in range(num_steps + 1):
            t = i / num_steps
            # Smooth easing (ease-in-out)
            t_smooth = t * t * (3 - 2 * t)
            angle = start_angle + (end_angle - start_angle) * t_smooth
            self._set_door_angle(door_name, self._door_marker_name, angle)
            time.sleep(1.0 / fps)

    def start_door_flapping_animation(self, marker_name):
        """Start the door animation with attached marker."""
        if self.animation_thread and self.animation_thread.is_alive():
            self.node.get_logger().warn('Animation already running')
            return
        
        # Create pose publishers
        self.door_pose_pub = self.node.create_publisher(PoseStamped, '/door_world_pose', 10)
        self.marker_pose_pub = self.node.create_publisher(PoseStamped, '/marker_world_pose', 10)
        
        # Marker offset from door center: front face offset + surface offset + user-specified offset
        # Front face is at y = -wall_thickness/2 from door center, plus 5mm outside
        marker_surface_offset = -0.005  # 5mm outside the door
        user_offset = np.array(self.door_marker_local_pos) if self.door_marker_local_pos is not None else np.array([0, 0, 0])
        local_marker_pos = np.array([0, -self.wall_thickness/2 + marker_surface_offset, 0]) + user_offset
        
        # Use stored door name, default to "front" for spawn_complete() compatibility
        door_name = self._door_name if self._door_name else "front"
        
        self.running = True
        self.animation_thread = threading.Thread(
            target=self._animate_door,
            args=(door_name, marker_name, local_marker_pos),
            daemon=True
        )
        self.animation_thread.start()
        self.node.get_logger().info('Door animation started')

    def stop_animation(self):
        """Stop the door animation."""
        self.running = False
        if self.animation_thread:
            self.animation_thread.join(timeout=1.0)
        self.node.get_logger().info('Door animation stopped')

    def spin(self):
        """Convenience method to spin the node."""
        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        """Clean shutdown of the printer and all resources."""
        self.cleanup()
        if self._bridge_proc:
            self._bridge_proc.terminate()
            self._bridge_proc.wait()
        if self._owns_node:
            self.node.destroy_node()

    def cleanup(self):
        """Remove all spawned entities."""
        self.stop_animation()
        for name in self.spawned_entities:
            self._delete_entity(name)
        self.spawned_entities.clear()
        self.markers.clear()
        self.walls.clear()


def main():
    """Simple main function demonstrating the printer usage."""
    rclpy.init()
    
    # Simple usage - just create and spawn with defaults
    printer = Simulated3DPrinter()
    printer.spawn_fast()
    
    # Demonstrate opening and closing the door with animation
    num_cycles = 3
    pause_between = 0.5  # seconds to pause between open/close
    animation_duration = 1.0  # seconds for each open/close animation
    #time.sleep(3)
    for i in range(num_cycles):
        printer.node.get_logger().info(f'Cycle {i + 1}/{num_cycles}')
        
        time.sleep(pause_between)
        printer.open_door(duration=animation_duration)
        
        time.sleep(pause_between)
        printer.close_door(duration=animation_duration)
    
    printer.node.get_logger().info('Door demo complete, spinning...')
    printer.spin()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
