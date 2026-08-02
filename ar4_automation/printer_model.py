"""Load a primitive-box printer approximation and emit it for Gazebo / MoveIt.

The box model is produced offline by tools/printer_mesh_to_boxes.py from a real
printer mesh and stored at models/printers/<name>.json. Boxes live in the
printer-LOCAL frame (origin at the printer's geometric center, +Z up, axes
aligned with the printer at zero yaw) — the same frame Simulated3DPrinter places
its walls in, so they drop straight into the spawn SDF.

One box list, two consumers:
  * to_sdf_links()  -> inline <visual>/<collision> for the Gazebo spawn (now)
  * boxes_in_base() -> base_link-frame boxes for MoveIt add_collision_box (later)

Keeping both on the same source means what you see in Gazebo is exactly what the
planner will avoid.
"""

import json
import os

import numpy as np

# models/printers lives at the repo root, one level above this package
DEFAULT_PRINTERS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'printers')


class PrinterModel:
    def __init__(self, name, footprint, boxes, bed_size=None, markers=None,
                 plate=None, procedures=None, meta=None):
        self.name = name
        self.footprint = footprint          # dict: width/depth/height (m)
        self.boxes = boxes                  # list of {center:[3], size:[3], rpy:[3]}
        self.bed_size = bed_size            # [w, d] (m) or None
        # this printer's removable build plate, and how the arm holds it:
        #   {size: [w, d, thickness], offset_pos: [3], offset_rpy: [3]}
        # offset_* is the plate CENTER's 6DOF pose in the GRIPPER (eef) frame
        # while grasped — see held_plate.plate_spec for defaults when absent.
        self.plate = plate
        # THE hand-programmed part of a printer: its pickup/place/scrape
        # waypoint lists, in the marker's local frame —
        #   {'frame': 'grasp', 'pickup': [...], 'place': [...], 'scrape': [...]}
        # 'frame': 'grasp' means every 'pos' entry is the pose of the gripper
        # TCP (robot_config 'grasp_frame'), so the SAME list runs on any arm;
        # the robot-specific flange offset is back-solved from TF at runtime
        # (printerAutomation._move_to_marker_offset). Entry kinds are the ones
        # documented at printerAutomation.offset_configs. NB 'traj' entries
        # replay recorded JOINT paths and are inherently robot-specific.
        self.procedures = procedures
        # ArUco mount poses in the printer-local frame, placed by CTRL+clicking
        # the box model in tools/view_printer_model.py:
        #   [{name, size, pos:[3], rpy:[3]}, ...]
        # rpy follows the same convention as the legacy door/static markers:
        # the plate's local +X is the INWARD normal, its +Z is up.
        # Which ArUco ID a mount wears is deliberately NOT stored here — one
        # model can be spawned several times with different IDs, so the texture
        # is a spawn-time argument (Simulated3DPrinter.marker_ids).
        self.markers = markers or []
        self.meta = meta or {}

    # ---- construction ----

    @classmethod
    def load(cls, name, printers_dir=None):
        """Load models/printers/<name>.json (name may also be a direct path)."""
        path = name if os.path.isfile(name) else os.path.join(
            printers_dir or DEFAULT_PRINTERS_DIR, f'{name}.json')
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"printer model '{name}' not found at {path}. Generate it with "
                f"tools/printer_mesh_to_boxes.py --printer {name} --mesh <mesh>")
        with open(path, 'r') as f:
            d = json.load(f)
        return cls(
            name=d.get('name', name),
            footprint=d['footprint'],
            boxes=d['boxes'],
            bed_size=d.get('bed_size'),
            markers=d.get('markers'),
            plate=d.get('plate'),
            procedures=d.get('procedures'),
            meta={k: d[k] for k in ('source_mesh', 'coverage', 'frame') if k in d},
        )

    # ---- footprint accessors (feed the door/marker placement math) ----

    @property
    def width(self):
        return float(self.footprint['width'])

    @property
    def depth(self):
        return float(self.footprint['depth'])

    @property
    def height(self):
        return float(self.footprint['height'])

    # ---- Gazebo (now) ----

    def entry_zone_indices(self):
        """Indices of boxes flagged "entry_zone": true in the JSON (marked with
        SHIFT+click in tools/definePrinterApproximateModel.py) — the ENTRY
        ZONE: the subset of boxes whose collisions a pickup sequence toggles
        off while the gripper is inside the printer and back on afterwards.
        Distinct from the global COLLISIONS switch, which covers everything."""
        return [i for i, b in enumerate(self.boxes) if b.get('entry_zone')]

    def to_sdf_links(self, include_collision=True, rgba=(0.5, 0.5, 0.5, 1.0),
                     entry_zone_collide=True,
                     entry_zone_open_rgba=(0.85, 0.65, 0.15, 0.6)):
        """SDF <visual> (and optional <collision>) blocks for every box, to be
        embedded inside a single <link>. Inline box geometry — no mesh files, so
        nothing depends on GZ_SIM_RESOURCE_PATH at spawn time.

        entry_zone_collide is the RUNTIME state of the ENTRY-ZONE boxes
        ("entry_zone": true): True (default) renders them like every other
        box, colliding; False (zone open) drops their <collision> blocks and
        recolors them entry_zone_open_rgba, so Gazebo shows exactly which
        volumes the arm is currently allowed inside.
        Simulated3DPrinter.set_entry_zone_collisions respawns the model with
        this flipped as the pickup sequence progresses."""
        r, g, b, a = rgba
        tr, tg, tb, ta = entry_zone_open_rgba
        out = ""
        for i, box in enumerate(self.boxes):
            solid = entry_zone_collide or not box.get('entry_zone')
            cx, cy, cz = box['center']
            sx, sy, sz = box['size']
            rr, pp, yy = box.get('rpy', [0.0, 0.0, 0.0])
            pose = f"{cx} {cy} {cz} {rr} {pp} {yy}"
            geom = f"<box><size>{sx} {sy} {sz}</size></box>"
            cr, cg, cb, ca = (r, g, b, a) if solid else (tr, tg, tb, ta)
            out += f"""
      <visual name="body_box_{i}">
        <pose>{pose}</pose>
        <geometry>{geom}</geometry>
        <material>
          <ambient>{cr} {cg} {cb} {ca}</ambient>
          <diffuse>{cr} {cg} {cb} {ca}</diffuse>
        </material>
      </visual>"""
            if include_collision and solid:
                out += f"""
      <collision name="body_box_{i}_collision">
        <pose>{pose}</pose>
        <geometry>{geom}</geometry>
      </collision>"""
        return out

    # ---- markers ----

    def get_marker(self, name):
        """The named mount from the JSON, or None."""
        for m in self.markers:
            if m.get('name') == name:
                return m
        return None

    def marker_aruco_rotation(self, name):
        """Rotation of a mount in the printer frame, in ArUco convention (Z out
        of the face) rather than the SDF plate convention (+X the inward
        normal). Returns a scipy Rotation, or None if the mount isn't there.

        The two frames are a fixed relabelling of axes:
            aruco X = -plate Y,  aruco Y = +plate Z,  aruco Z = -plate X
        i.e. aruco Z is the face's OUTWARD normal. For a marker on the front
        (-Y) face — plate yawed +90 deg — that gives X=[1,0,0], Z=[0,-1,0],
        which is what the old hardcoded front-face math produced."""
        from scipy.spatial.transform import Rotation as R

        mk = self.get_marker(name)
        if mk is None:
            return None
        plate_to_aruco = np.array([[0.0, 0.0, -1.0],
                                   [-1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0]])
        return R.from_euler('xyz', mk['rpy']) * R.from_matrix(plate_to_aruco)

    def pose_from_marker(self, name, marker_pos, marker_euler):
        """Where the printer must be for its named mount to sit at an observed
        marker pose. The inverse of the usual direction: a detected marker plus
        a known mount offset pins the printer body.

        marker_pos/marker_euler are the ArUco pose in base_link, in the frames
        ArucoDetector reports (euler is scipy intrinsic 'XYZ', matching
        printer_automation's saved state and marker_sources.load_marker_poses).

        Returns (pos, orient) ready for Simulated3DPrinter, where orient is
        extrinsic xyz — the convention its tf quaternion_from_euler uses.
        Returns None if the model has no such mount."""
        from scipy.spatial.transform import Rotation as R

        mk = self.get_marker(name)
        R_pm = self.marker_aruco_rotation(name)
        if mk is None or R_pm is None:
            return None

        R_bm = R.from_euler('XYZ', np.asarray(marker_euler, dtype=float))
        R_bp = R_bm * R_pm.inv()                       # printer -> base
        pos = np.asarray(marker_pos, dtype=float) - R_bp.apply(mk['pos'])
        return [float(v) for v in pos], [float(v) for v in R_bp.as_euler('xyz')]

    # ---- MoveIt planning scene (deferred; the forward-compat seam) ----

    def boxes_in_base(self, pos, quat_xyzw, id_prefix="printer", subset='all'):
        """Transform local boxes into the base frame given the printer's
        world/base pose. Returns dicts ready for
        MoveIt2.add_collision_box(id=.., size=.., position=.., quat_xyzw=..):
            [{'id', 'size', 'position', 'quat_xyzw'}, ...]

        subset selects which boxes: 'all' (default — the initial planning
        scene, entry-zone boxes included since they collide by default),
        'entry_zone' (just that runtime-switchable subset, for
        set_entry_zone_collisions to remove/re-add mid-sequence), or 'static'
        (everything else). Ids are numbered by JSON index regardless of
        subset, so the same box always maps to the same scene object.

        Each box's base orientation is the printer orientation composed with the
        box's local rpy; its base position is the printer pose applied to the
        local center."""
        from scipy.spatial.transform import Rotation as R

        R_pb = R.from_quat(list(quat_xyzw))     # printer -> base
        pos = np.asarray(pos, dtype=float)
        result = []
        for i, box in enumerate(self.boxes):
            in_zone = bool(box.get('entry_zone'))
            if subset == 'entry_zone' and not in_zone:
                continue
            if subset == 'static' and in_zone:
                continue
            c = np.asarray(box['center'], dtype=float)
            rpy = box.get('rpy', [0.0, 0.0, 0.0])
            base_pos = pos + R_pb.apply(c)
            R_box = R_pb * R.from_euler('xyz', rpy)
            result.append({
                'id': f"{id_prefix}_{self.name}_box{i}",
                'size': [float(s) for s in box['size']],
                'position': [float(x) for x in base_pos],
                'quat_xyzw': [float(x) for x in R_box.as_quat()],
            })
        return result
