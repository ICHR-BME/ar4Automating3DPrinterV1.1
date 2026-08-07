"""Spawn a real-printer box model, with its ArUco mounts, into Gazebo.

One printer == one SDF model: the primitive boxes from
models/printers/<name>.json (visual AND collision, so what you see is what
MoveIt will avoid) plus a textured plate for every mount in that JSON's
"markers" list, placed with tools/view_printer_model.py.

collide=False drops the <collision> blocks, leaving visual-only boxes the arm
passes straight through — the Gazebo half of the runner scripts' COLLISIONS
switch, which it follows by default via node.collision_scene_enabled.

Mounts are the single source of truth for where a marker sits on a printer:
  * spawn()             puts the plate there in Gazebo
  * marker_pose_in_base() says where the camera should therefore see it
  * PrinterModel.pose_from_marker() (via from_marker) inverts it: an observed
    marker pins the printer body
so a marker moved in the JSON moves in all three at once.

The legacy hollow-box printer — five walls, a separately spawned swinging door
and door marker, the animation thread, and the hardcoded front-face marker
math — is gone. Every spawn now goes through a printer model.
"""

import os
import atexit
import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
import tf_transformations
from geometry_msgs.msg import Point, Quaternion
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose, SpawnEntity

from .printer_model import PrinterModel


class GzEntityClient:
    """Persistent gz-service plumbing shared by everything that puts models
    into Gazebo: spawn/delete entities and move them with set_pose, all through
    long-lived ros_gz_bridge service bridges instead of a fresh
    `ros2 run ros_gz_sim create` executable per call.

    Simulated3DPrinter spawns through this; HeldPlateVisual (held_plate.py)
    spawns AND re-poses through it every follow tick."""

    def __init__(self, node, world_name='default'):
        self.node = node
        self.world_name = world_name
        self.spawn_client = None
        self.pose_client = None
        self._bridge_proc = None
        self._pose_bridge_proc = None
        self.spawned_entities = []

    def _setup_spawn_client(self):
        """Bring up ONE persistent ros_gz_bridge for the create service, plus its
        ROS client. Spawning through this in-process client — instead of a fresh
        `ros2 run ros_gz_sim create` executable per model — is what makes spawns
        fast and reliable.

        Reuse an already-running bridge when possible: every leaked
        parameter_bridge subprocess registers as the SAME node name
        (/ros_gz_bridge), and duplicate node names corrupt DDS discovery —
        symptom: fresh nodes whose camera_info subscription silently never
        matches, so aruco detection runs uncalibrated and spots nothing."""
        create_srv = f'/world/{self.world_name}/create'
        self.spawn_client = self.node.create_client(SpawnEntity, create_srv)
        if self.spawn_client.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().info('gz create service available (reusing bridge)')
            return

        # set_pose rides along on the same bridge, so a HeldPlateVisual in the
        # same session finds it already bridged (it starts its own only when a
        # reused older bridge lacks it)
        self._bridge_proc = subprocess.Popen(
            ['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
             f'{create_srv}@ros_gz_interfaces/srv/SpawnEntity',
             f'/world/{self.world_name}/set_pose@ros_gz_interfaces/srv/SetEntityPose',
             '--ros-args', '-r', f'__node:=gz_srv_bridge_{os.getpid()}'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        atexit.register(self._bridge_proc.terminate)
        self.node.get_logger().info('Started ros_gz_bridge for the create service')
        if self.spawn_client.wait_for_service(timeout_sec=10.0):
            self.node.get_logger().info('gz create service available')
        else:
            self.node.get_logger().warn('gz create service not available')

    def _setup_pose_client(self):
        """Client for /world/<world>/set_pose. Reuses a bridge that already
        carries it (ours, or one _setup_spawn_client started); only when none
        does is a dedicated pose bridge launched. Returns True when the service
        is ready."""
        set_srv = f'/world/{self.world_name}/set_pose'
        if self.pose_client is None:
            self.pose_client = self.node.create_client(SetEntityPose, set_srv)
        if self.pose_client.wait_for_service(timeout_sec=2.0):
            return True

        self._pose_bridge_proc = subprocess.Popen(
            ['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
             f'{set_srv}@ros_gz_interfaces/srv/SetEntityPose',
             '--ros-args', '-r', f'__node:=gz_pose_bridge_{os.getpid()}'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        atexit.register(self._pose_bridge_proc.terminate)
        self.node.get_logger().info('Started ros_gz_bridge for the set_pose service')
        if self.pose_client.wait_for_service(timeout_sec=10.0):
            return True
        self.node.get_logger().warn('gz set_pose service not available')
        return False

    def set_entity_pose_async(self, name, pos, quat_xyzw):
        """Move a spawned model (world frame), fire-and-forget. Returns the
        service future, or None when the set_pose client isn't ready — callers
        poll future.done() to rate-limit instead of queueing calls."""
        if self.pose_client is None or not self.pose_client.service_is_ready():
            return None
        req = SetEntityPose.Request()
        req.entity.name = name
        req.entity.type = Entity.MODEL
        req.pose.position = Point(
            x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        req.pose.orientation = Quaternion(
            x=float(quat_xyzw[0]), y=float(quat_xyzw[1]),
            z=float(quat_xyzw[2]), w=float(quat_xyzw[3]))
        return self.pose_client.call_async(req)

    def _wait_future(self, future, timeout_sec=10.0):
        """Block until a service future completes, spinning the node only when
        no background executor owns it (mirrors _spin_or_sleep's rule)."""
        start = time.time()
        while not future.done() and (time.time() - start) < timeout_sec:
            self._spin_or_sleep(0.05)
        return future.done()

    def _spin_or_sleep(self, timeout_sec):
        """Process callbacks for ~timeout_sec. Never spin_once() a node a
        background executor owns (it detaches the node and kills callback
        delivery for good); just sleep in that case."""
        if self.node.executor is None:
            rclpy.spin_once(self.node, timeout_sec=timeout_sec)
        else:
            time.sleep(timeout_sec)

    def _spawn_entity(self, sdf_str, name, pos, rpy):
        """Spawn one model from an inline SDF string via the persistent create
        client. Returns True on success. Inline SDF (no temp file) means texture
        paths inside the SDF must be absolute — see _abs_texture."""
        if self.spawn_client is None or not self.spawn_client.service_is_ready():
            self.node.get_logger().error(f'create service not ready; cannot spawn {name}')
            return False
        req = SpawnEntity.Request()
        ef = req.entity_factory
        ef.name = name
        ef.sdf = sdf_str
        ef.relative_to = 'world'
        q = tf_transformations.quaternion_from_euler(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        ef.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        ef.pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
        future = self.spawn_client.call_async(req)
        if not self._wait_future(future):
            self.node.get_logger().warn(f'spawn of {name} timed out')
            return False
        res = future.result()
        if res is not None and res.success:
            self.node.get_logger().info(f'Spawned {name}')
            return True
        self.node.get_logger().warn(f'spawn of {name} reported failure')
        return False

    def _entity_exists(self, name):
        """True if a model of this name is in the world right now.

        None if the query itself failed, so callers can tell 'definitely gone'
        from 'no idea' and not wait on a world they can't see."""
        try:
            result = subprocess.run(
                ['gz', 'model', '--list'],
                capture_output=True, text=True, timeout=5.0)
        except (subprocess.SubprocessError, OSError):
            return None
        if result.returncode != 0:
            return None
        return any(line.strip() == f'- {name}'
                   for line in result.stdout.splitlines())

    def _delete_entity(self, name, wait_timeout=3.0):
        """Remove an entity from Gazebo and WAIT until it is really gone.

        The remove service returns as soon as gz accepts the request, but the
        entity isn't erased until the end of an update cycle. Spawning the same
        name immediately after therefore races: the create lands first, then the
        pending removal deletes the model that was just made, and the spawn
        looks successful while nothing appears in the world. That is why
        respawning a printer used to silently drop whichever body already
        existed. Polling until the name disappears removes the race."""
        if self._entity_exists(name) is False:
            return          # nothing to remove, so nothing to wait for

        delete_cmd = [
            'gz', 'service', '-s', f'/world/{self.world_name}/remove',
            '--reqtype', 'gz.msgs.Entity', '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000', '--req', f'name: "{name}" type: MODEL'
        ]
        try:
            result = subprocess.run(delete_cmd, capture_output=True, text=True, check=True)
            msg = result.stdout.strip() or f'{name} deleted successfully'
            self.node.get_logger().debug(f'Delete: {msg}')
        except subprocess.CalledProcessError:
            pass  # entity may not exist
        except Exception as e:
            self.node.get_logger().warn(f'Delete exception for {name}: {e}')
            return

        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            if self._entity_exists(name) is not True:
                return
            time.sleep(0.05)
        self.node.get_logger().warn(
            f'{name} still present {wait_timeout}s after remove; spawning anyway '
            f'(it may be deleted again by the pending removal)')

    def _terminate_bridges(self):
        for proc in (self._bridge_proc, self._pose_bridge_proc):
            if proc:
                proc.terminate()
                proc.wait()
        self._bridge_proc = None
        self._pose_bridge_proc = None

    def cleanup(self):
        """Remove all spawned entities."""
        for name in self.spawned_entities:
            self._delete_entity(name)
        self.spawned_entities.clear()


class Simulated3DPrinter(GzEntityClient):
    """A printer model spawned into Gazebo at a known base_link pose.

    printer_model: name in models/printers/ (or a PrinterModel).
    marker_ids: which ArUco ID each mount wears, {mount_name: id} — e.g.
        {'door': 2}. A bare int is shorthand for the default mount. Mounts left
        out of the dict are not drawn (no texture to give them).
    """

    # models/aruco_marker/ (marker textures) lives at the repo root
    DEFAULT_MODEL_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models', 'aruco_marker', '')

    DEFAULT_PRINTER_MODEL = 'a1_mini'
    DEFAULT_MOUNT = 'door'
    MARKER_TEXTURE = 'materials/textures/marker6x6_{}.png'

    def __init__(
        self,
        node=None,
        pos=(0.6, 0.0, 0.2),
        orient=(0.0, 0.0, np.pi),
        printer_model=None,
        marker_ids=None,
        marker_size=None,
        marker_texture=None,
        model_dir=None,
        use_bad_frame=True,
        world_name='default',
        name=None,
        collide=None,
    ):
        self._owns_node = node is None
        super().__init__(node=node if node else Node('simulated_printer'),
                         world_name=world_name)

        # Gazebo solidity. The model is <static>, so its boxes never move, but
        # they are real collision geometry: the arm cannot pass through them, it
        # STALLS against them — the joints fight a wall they can't push, the
        # trajectory falls behind, and the controller aborts. That looks nothing
        # like a planner refusing a goal, so a MoveIt-only switch is not enough
        # to get the arm out of a printer's way.
        # None = follow the node's collision_scene_enabled, so the runner
        # scripts' COLLISIONS var turns off the planning scene AND the physics
        # together. False spawns visual-only boxes (see-through in Gazebo);
        # <static> means nothing falls as a result. Marker plates are visuals
        # with no <collision> either way, so they never block anything.
        self.collide = (bool(getattr(self.node, 'collision_scene_enabled', True))
                        if collide is None else bool(collide))
        # runtime state of the model's ENTRY-ZONE boxes ("entry_zone": true
        # in the JSON, flagged via tools/definePrinterApproximateModel.py):
        # True = zone closed, those boxes collide like the rest; False = a
        # pickup sequence has opened the zone (recolored, no physics).
        # Flipped by set_entry_zone_collisions.
        self.entry_zone_collide = True

        pos = np.array(pos, dtype=float)
        orient = np.array(orient, dtype=float)

        # good frame -> bad frame (base_link/world). Only the frame rotation,
        # not the EEF offset angles that to_bad_frame adds.
        if use_bad_frame and hasattr(self.node, 'frameRotationAngles'):
            from scipy.spatial.transform import Rotation as R_scipy
            R_BF_GF = R_scipy.from_euler("XYZ", self.node.frameRotationAngles, degrees=False)
            R_GF_BF = R_BF_GF.inv()
            pos = R_GF_BF.apply(pos)
            # extrinsic 'xyz' both ways: self.q below builds the quaternion with
            # tf's sxyz, so reading the spec as intrinsic "XYZ" here silently
            # reinterpreted the result as extrinsic. Single-axis specs (every
            # xArm one) are identical under both, but the AR4's multi-axis
            # specs came out 120 deg / 0.37 m wrong.
            R_orient = R_scipy.from_euler("xyz", orient, degrees=False)
            orient = (R_GF_BF * R_orient).as_euler("xyz", degrees=False)

        self.pos = np.array(pos, dtype=float)
        self.orient = np.array(orient, dtype=float)

        model = printer_model if printer_model is not None else self.DEFAULT_PRINTER_MODEL
        self.printer_model = (model if isinstance(model, PrinterModel)
                              else PrinterModel.load(model))

        self.width = self.printer_model.width
        self.depth = self.printer_model.depth
        self.height = self.printer_model.height

        self.marker_ids = self._normalize_marker_ids(marker_ids)
        self.marker_size = marker_size
        self.marker_texture = marker_texture or self.MARKER_TEXTURE
        self.model_dir = model_dir or self.DEFAULT_MODEL_DIR

        # entity name: model + the IDs it wears, so two printers of the same
        # model don't overwrite each other in the world
        ids = '_'.join(str(v) for _, v in sorted(self.marker_ids.items()))
        self.name = name or f"printer_{self.printer_model.name}" + (f"_{ids}" if ids else "")

        # spawn orientation as tf 'sxyz' (extrinsic xyz) — the same quaternion
        # marker_pose_in_base() reasons with, so an estimate can't disagree with
        # where Gazebo actually put the printer
        self.q = [float(x) for x in tf_transformations.quaternion_from_euler(
            float(self.orient[0]), float(self.orient[1]), float(self.orient[2]))]

    # ---- construction ----

    @classmethod
    def from_marker(cls, marker_pos, marker_euler, printer_model,
                    mount=None, **kwargs):
        """Place a printer so its `mount` lands on an observed marker pose.

        marker_pos/marker_euler are in base_link, in ArucoDetector's frames
        (euler = scipy intrinsic 'XYZ') — i.e. straight out of
        marker_sources.load_marker_poses(). The pose already IS in the bad
        frame, so no frame rotation is applied to it.

        Raises ValueError if the model has no such mount (place one with
        tools/view_printer_model.py).
        """
        model = (printer_model if isinstance(printer_model, PrinterModel)
                 else PrinterModel.load(printer_model))
        mount = mount or cls.DEFAULT_MOUNT
        derived = model.pose_from_marker(mount, marker_pos, marker_euler)
        if derived is None:
            raise ValueError(
                f"printer model '{model.name}' has no '{mount}' mount — place one "
                f"with tools/view_printer_model.py")
        pos, orient = derived
        kwargs.setdefault('use_bad_frame', False)
        return cls(pos=pos, orient=orient, printer_model=model, **kwargs)

    def _normalize_marker_ids(self, marker_ids):
        """{mount_name: aruco_id}, accepting a bare int for the default mount."""
        if marker_ids is None:
            return {}
        if isinstance(marker_ids, (int, np.integer)):
            mount = (self.DEFAULT_MOUNT
                     if self.printer_model.get_marker(self.DEFAULT_MOUNT)
                     else (self.printer_model.markers[0]['name']
                           if self.printer_model.markers else self.DEFAULT_MOUNT))
            return {mount: int(marker_ids)}
        return {str(k): int(v) for k, v in marker_ids.items()}

    # ---- gz plumbing shared with GzEntityClient; only the texture-path
    # helper (needs model_dir) stays here ----

    def _abs_texture(self, tex):
        """Absolute path for a texture referenced in an inline SDF (relative
        paths only resolve when the SDF is a file inside model_dir; inline SDF
        strings have no base dir)."""
        if not tex or os.path.isabs(tex):
            return tex
        return os.path.join(self.model_dir, tex)


    # ---- SDF ----

    def _marker_texture_path(self, mount_name):
        """Texture for a mount, or None when no ArUco ID was assigned to it."""
        marker_id = self.marker_ids.get(mount_name)
        if marker_id is None:
            return None
        return self._abs_texture(self.marker_texture.format(marker_id))

    def _generate_sdf(self, model_name):
        """One static SDF model: the printer's boxes plus a textured plate for
        every mount that was given an ArUco ID."""
        visuals = self.printer_model.to_sdf_links(
            include_collision=self.collide,
            entry_zone_collide=self.entry_zone_collide)

        for mk in self.printer_model.markers:
            texture = self._marker_texture_path(mk.get('name'))
            if not texture:
                continue
            # the mount pose is already in the printer-local frame with the
            # plate convention baked in (+X the inward normal, +Z up), so it
            # goes straight into <pose> — no anchor math
            mx, my, mz = mk['pos']
            mr, mp, myaw = mk['rpy']
            msize = self.marker_size or mk.get('size', 0.05)
            visuals += f"""
      <visual name="marker_{mk.get('name', 'marker')}">
        <pose>{mx} {my} {mz} {mr} {mp} {myaw}</pose>
        <geometry>
          <box>
            <size>0.0001 {msize} {msize}</size>
          </box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
          <pbr>
            <metal>
              <albedo_map>{texture}</albedo_map>
              <metalness>0.0</metalness>
              <roughness>1.0</roughness>
            </metal>
          </pbr>
        </material>
      </visual>"""

        return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="body">
{visuals}
    </link>
  </model>
</sdf>"""

    # ---- spawn ----

    def spawn(self, name=None):
        """Spawn the printer (boxes + marker plates) as a single gz model.

        No TF warmup and no per-part poses: everything is a visual inside the
        body model, so the only pose that matters is self.pos/self.orient.
        """
        name = name or self.name
        self._setup_spawn_client()
        self._delete_entity(name)
        ok = self._spawn_entity(self._generate_sdf(name), name, self.pos, self.orient)
        if ok:
            if name not in self.spawned_entities:
                self.spawned_entities.append(name)
            n_boxes = len(self.printer_model.boxes)
            n_off = sum(1 for b in self.printer_model.boxes
                        if not b.get('collide', True))
            if self.collide:
                solidity = (f"solid, {n_off} flagged visual-only" if n_off
                            else "solid")
            else:
                solidity = "VISUAL-ONLY, arm passes through"
            self.node.get_logger().info(
                f"Spawned {name}: {self.printer_model.name}, "
                f"{n_boxes} boxes ({solidity}), "
                f"markers {self.marker_ids} "
                f"at {[round(float(v), 4) for v in self.pos]} "
                f"rpy {[round(float(v), 4) for v in self.orient]}")
        return ok

    def set_entry_zone_collisions(self, enabled):
        """Toggle the collisions of the model's ENTRY-ZONE boxes in GAZEBO by
        respawning the printer with those boxes' collision blocks (and their
        color) flipped — a spawned model's geometry can't be edited in place.
        Pose, markers, and every other box are identical across the respawn.
        No-op when nothing changed or the model has no entry-zone boxes;
        True on success (state-only, without a respawn, if not spawned yet)."""
        enabled = bool(enabled)
        if enabled == self.entry_zone_collide:
            return True
        if not self.printer_model.entry_zone_indices():
            return True
        self.entry_zone_collide = enabled
        if self.name not in self.spawned_entities:
            return True          # hardware run / not spawned: state only
        ok = self.spawn(self.name)
        if ok:
            self.node.get_logger().info(
                f"{self.name}: entry zone "
                f"{'CLOSED (colliding again)' if enabled else 'OPEN (recolored, no physics)'}")
        return ok

    # ---- where its markers are ----

    def marker_pose_in_base(self, mount=None):
        """A mount's ArUco pose in base_link (Z out of the face), matching
        ArucoDetector's frames: (pos, euler) with euler scipy intrinsic 'XYZ'.

        The exact inverse of PrinterModel.pose_from_marker, so registering this
        as an estimate and then back-solving the printer from a perfect
        detection returns the printer to where it was spawned. World ==
        base_link here (both are the Gazebo fixed frame).

        Raises ValueError if the model has no such mount.
        """
        from scipy.spatial.transform import Rotation as R_scipy

        mount = mount or self.DEFAULT_MOUNT
        mk = self.printer_model.get_marker(mount)
        R_pm = self.printer_model.marker_aruco_rotation(mount)
        if mk is None or R_pm is None:
            raise ValueError(
                f"printer model '{self.printer_model.name}' has no '{mount}' mount — "
                f"place one with tools/view_printer_model.py")

        R_bp = R_scipy.from_quat(self.q)          # printer -> base
        pos = np.asarray(self.pos, dtype=float) + R_bp.apply(mk['pos'])
        euler = (R_bp * R_pm).as_euler('XYZ', degrees=False)
        return pos, euler

    def register_marker_estimates(self, node=None, mounts=None):
        """Register each mount's pose as an initial marker estimate on the node,
        so a scan knows where to look. Returns the marker IDs registered."""
        node = node or self.node
        registered = []
        for mount, marker_id in self.marker_ids.items():
            if mounts is not None and mount not in mounts:
                continue
            bad_pos, bad_euler = self.marker_pose_in_base(mount)
            node.register_estimated_marker(
                marker_id=marker_id, bad_pos=bad_pos, bad_euler=bad_euler)
            registered.append(marker_id)
        return registered

    # ---- lifecycle ----

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
        self._terminate_bridges()
        if self._owns_node:
            self.node.destroy_node()


def main():
    """Spawn one printer of each model with markers 1 and 2, then spin."""
    rclpy.init()
    node = Node('simulated_printer_demo')
    for model, marker_id, y in (('a1', 1, 0.35), ('a1_mini', 2, -0.35)):
        printer = Simulated3DPrinter(
            node=node, printer_model=model, marker_ids={'door': marker_id},
            pos=[0.6, y, PrinterModel.load(model).height / 2.0],
            orient=[0.0, 0.0, 0.0], use_bad_frame=False,
        )
        printer.spawn()
        node.get_logger().info(
            f"{model} door marker {marker_id} should be seen at "
            f"{np.round(printer.marker_pose_in_base('door')[0], 3)}")
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
