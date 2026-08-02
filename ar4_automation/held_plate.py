"""Visualize the build plate the gripper is holding, as a Gazebo box that
tracks the end effector.

After a pickup the arm is really carrying a plate, so its swept volume is
bigger than the bare gripper. This module makes that visible: attach() spawns
a plate-sized box and a node timer then re-poses it every tick at

    T_world_plate = T_base_grasp (from TF)  *  T_grasp_plate (the 6DOF offset)

so it follows the gripper through every motion. detach() removes it.

The GRASP frame is the gripper's TCP (robot_config 'grasp_frame' —
'link_tcp' on the xarm robots, at the FINGERTIPS, 0.172 m past link_eef on
the xArm 6), not the eef flange: anchored at link_eef the plate sat inside
the gripper body. Robots without a TCP frame (AR4) fall back to the eef link
and must bake the gripper length into offset_pos.

The gz box is VISUAL-ONLY on purpose — real <collision> geometry would wedge
the arm against its own plate. World == base_link here, the same assumption
every printer spawn already makes.

Which plate: plate_spec() reads the printer model JSON (models/printers/
<name>.json). An explicit "plate" section wins:

    "plate": {"size": [w, d, thickness],
              "offset_pos": [x, y, z],        # plate CENTER in the grasp frame
              "offset_rpy": [r, p, y]}        # euler XYZ (intrinsic)

Absent entries fall back to bed_size + DEFAULT_PLATE_THICKNESS and a generic
grasp: plate hanging out of the jaws along tool +Z, faces across the jaws,
near EDGE exactly at the grasp-frame origin. Tune the real grasp per printer
in its JSON.

Path planning: printerAutomation.attach_held_plate also publishes the same
box as a MoveIt ATTACHED collision object on the grasp frame (touch_links =
the gripper links), so the planner carries the plate through every plan while
it's attached. One source for size + offset feeds both, so the Gazebo visual
and the planning box can never disagree.

Not wired into pickupPlate/scrapePlate yet; drive it via
printerAutomation.attach_held_plate / detach_held_plate (see
testHeldPlateVisual.py).
"""

import numpy as np
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R

from .printer_model import PrinterModel
from .simulated3DPrinter import GzEntityClient

# glass/PEI sheets are 2-4 mm; the box only needs to read as "a plate"
DEFAULT_PLATE_THICKNESS = 0.003


def plate_spec(printer_model):
    """The build plate a printer type hands the arm: size + grasp offset.

    printer_model: name in models/printers/ or a PrinterModel.
    Returns {'size': [w, d, t], 'offset_pos': [3], 'offset_rpy': [3]} with the
    offset convention from the module docstring (plate center in the eef
    frame). Raises ValueError when the model records neither a plate nor a
    bed_size to derive one from.
    """
    model = (printer_model if isinstance(printer_model, PrinterModel)
             else PrinterModel.load(printer_model))
    plate = dict(model.plate or {})

    if 'size' in plate:
        size = [float(v) for v in plate['size']]
    elif model.bed_size:
        size = [float(model.bed_size[0]), float(model.bed_size[1]),
                DEFAULT_PLATE_THICKNESS]
    else:
        raise ValueError(
            f"printer model '{model.name}' has no 'plate' section and no "
            f"'bed_size' — add one to its JSON in models/printers/")

    # default grasp: rolling +90 deg about grasp-frame X sends the plate's
    # thickness axis (local Z) across the jaws and its depth axis (local Y)
    # along tool +Z, i.e. the plate sticks straight out of the gripper;
    # centering it depth/2 out puts its near EDGE exactly at the grasp-frame
    # origin (the fingertips on robots with a link_tcp)
    offset_pos = [float(v) for v in plate.get('offset_pos',
                                              [0.0, 0.0, size[1] / 2.0])]
    offset_rpy = [float(v) for v in plate.get('offset_rpy',
                                              [np.pi / 2.0, 0.0, 0.0])]
    return {'size': size, 'offset_pos': offset_pos, 'offset_rpy': offset_rpy}


class HeldPlateVisual(GzEntityClient):
    """A grasped build plate rendered in Gazebo, glued to the gripper by TF.

    node must be a printerAutomation (it supplies tf_buffer, base_link_name,
    grasp_frame_name, and hosts the follow timer on its background executor).
    Size/offset come from plate_spec(printer_model); pass size/offset_pos/
    offset_rpy to override individual pieces, grasp_frame to anchor somewhere
    other than the robot's configured gripper TCP.
    """

    def __init__(self, node, printer_model=None, size=None, offset_pos=None,
                 offset_rpy=None, grasp_frame=None,
                 rgba=(0.9, 0.45, 0.1, 0.9),
                 name='held_plate', world_name='default', follow_rate_hz=15.0):
        super().__init__(node=node, world_name=world_name)
        self.grasp_frame = grasp_frame or getattr(
            node, 'grasp_frame_name', None) or node.end_effector_name
        self._warned_no_grasp_tf = False
        spec = (plate_spec(printer_model) if printer_model is not None
                else {'size': None, 'offset_pos': [0.0, 0.0, 0.0],
                      'offset_rpy': [0.0, 0.0, 0.0]})
        if size is None and spec['size'] is None:
            raise ValueError("HeldPlateVisual needs printer_model or size")
        self.size = [float(v) for v in (size if size is not None else spec['size'])]
        self.offset_pos = np.array(
            offset_pos if offset_pos is not None else spec['offset_pos'],
            dtype=float)
        self.offset_rpy = np.array(
            offset_rpy if offset_rpy is not None else spec['offset_rpy'],
            dtype=float)
        # same euler convention as the rest of the stack (scipy intrinsic XYZ)
        self._R_off = R.from_euler('XYZ', self.offset_rpy)
        self.rgba = rgba
        self.name = name
        self.follow_period = 1.0 / float(follow_rate_hz)
        self._timer = None
        self._pose_future = None
        self.attached = False

    # ---- pose math ----

    def _plate_pose_in_world(self):
        """(pos, quat_xyzw) of the plate center in world/base_link from the
        CURRENT grasp-frame TF, or None while TF has no transform yet.

        A missing grasp frame (gripper not in the URDF, so no link_tcp) falls
        back to the eef flange with a one-time warning — the plate then sits
        inside the gripper, but it still follows the arm."""
        try:
            tf = self.node.tf_buffer.lookup_transform(
                self.node.base_link_name, self.grasp_frame, Time())
        except Exception:
            if self.grasp_frame == self.node.end_effector_name:
                return None
            try:
                tf = self.node.tf_buffer.lookup_transform(
                    self.node.base_link_name, self.node.end_effector_name,
                    Time())
            except Exception:
                return None
            if not self._warned_no_grasp_tf:
                self._warned_no_grasp_tf = True
                self.node.get_logger().warn(
                    f"held plate: no TF for grasp frame '{self.grasp_frame}' — "
                    f"anchoring at {self.node.end_effector_name} instead (the "
                    f"plate will overlap the gripper body)")
        t, q = tf.transform.translation, tf.transform.rotation
        R_grasp = R.from_quat([q.x, q.y, q.z, q.w])
        pos = np.array([t.x, t.y, t.z]) + R_grasp.apply(self.offset_pos)
        return pos, (R_grasp * self._R_off).as_quat()

    # ---- SDF ----

    def _generate_sdf(self):
        """<static> so gravity never takes it — set_pose is the only mover.
        Visual-only: a solid plate would collide with the very printers the
        gripper carries it into."""
        w, d, t = self.size
        r, g, b, a = self.rgba
        return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{self.name}">
    <static>true</static>
    <link name="plate">
      <visual name="plate_visual">
        <geometry><box><size>{w} {d} {t}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

    # ---- attach / detach ----

    def attach(self):
        """Spawn the plate at the gripper's current pose and start following.
        Safe to call again after detach(); True on success."""
        if self.attached:
            return True
        self._setup_spawn_client()
        can_follow = self._setup_pose_client()
        if not can_follow:
            self.node.get_logger().warn(
                'held plate: set_pose unavailable — plate will spawn but NOT '
                'track the gripper')
        pose = self._plate_pose_in_world()
        if pose is None:
            self.node.get_logger().warn(
                'held plate: no grasp-frame TF yet; spawning at the origin '
                'until the follow timer sees one')
            pos, quat = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
        else:
            pos, quat = pose
        self._delete_entity(self.name)   # a stale one from a previous run
        rpy = R.from_quat(quat).as_euler('xyz')
        if not self._spawn_entity(self._generate_sdf(), self.name, pos, rpy):
            return False
        self.spawned_entities.append(self.name)
        self.attached = True
        if can_follow:
            self._timer = self.node.create_timer(
                self.follow_period, self._follow_tick)
        self.node.get_logger().info(
            f"held plate attached: {[round(v, 4) for v in self.size]} m on "
            f"'{self.grasp_frame}' at offset "
            f"{self.offset_pos.round(4).tolist()} / "
            f"rpy {self.offset_rpy.round(4).tolist()}")
        return True

    def detach(self):
        """Stop following and remove the plate from Gazebo."""
        if self._timer is not None:
            self._timer.cancel()
            self.node.destroy_timer(self._timer)
            self._timer = None
        self._pose_future = None
        if self.attached:
            self._delete_entity(self.name)
            if self.name in self.spawned_entities:
                self.spawned_entities.remove(self.name)
            self.attached = False
            self.node.get_logger().info('held plate detached')

    def _follow_tick(self):
        """One set_pose per tick, and only when the previous one landed —
        gz never sees a backlog, it just gets the freshest pose we have."""
        if self._pose_future is not None and not self._pose_future.done():
            return
        pose = self._plate_pose_in_world()
        if pose is None:
            return
        pos, quat = pose
        self._pose_future = self.set_entity_pose_async(self.name, pos, quat)

    # ---- MoveIt planning scene ----
    # The live hookup is printerAutomation.attach_held_plate, which publishes
    # this plate's size/offset as a box ATTACHED to the grasp frame
    # (moveit2.attach_collision_box) — link-relative, so move_group carries it
    # without any per-tick updates.

    def collision_box_in_base(self, object_id='held_plate'):
        """The plate as a MoveIt2.add_collision_box(**kwargs) dict at its
        CURRENT world pose, or None without TF — a static snapshot for e.g.
        planning around a plate that was set down, not the attached-object
        path above."""
        pose = self._plate_pose_in_world()
        if pose is None:
            return None
        pos, quat = pose
        return {'id': object_id, 'size': list(self.size),
                'position': [float(v) for v in pos],
                'quat_xyzw': [float(v) for v in quat]}
