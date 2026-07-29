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
    def __init__(self, name, footprint, boxes, bed_size=None, meta=None):
        self.name = name
        self.footprint = footprint          # dict: width/depth/height (m)
        self.boxes = boxes                  # list of {center:[3], size:[3], rpy:[3]}
        self.bed_size = bed_size            # [w, d] (m) or None
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

    def to_sdf_links(self, include_collision=True, rgba=(0.5, 0.5, 0.5, 1.0)):
        """SDF <visual> (and optional <collision>) blocks for every box, to be
        embedded inside a single <link>. Inline box geometry — no mesh files, so
        nothing depends on GZ_SIM_RESOURCE_PATH at spawn time."""
        r, g, b, a = rgba
        out = ""
        for i, box in enumerate(self.boxes):
            cx, cy, cz = box['center']
            sx, sy, sz = box['size']
            rr, pp, yy = box.get('rpy', [0.0, 0.0, 0.0])
            pose = f"{cx} {cy} {cz} {rr} {pp} {yy}"
            geom = f"<box><size>{sx} {sy} {sz}</size></box>"
            out += f"""
      <visual name="body_box_{i}">
        <pose>{pose}</pose>
        <geometry>{geom}</geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
        </material>
      </visual>"""
            if include_collision:
                out += f"""
      <collision name="body_box_{i}_collision">
        <pose>{pose}</pose>
        <geometry>{geom}</geometry>
      </collision>"""
        return out

    # ---- MoveIt planning scene (deferred; the forward-compat seam) ----

    def boxes_in_base(self, pos, quat_xyzw, id_prefix="printer"):
        """Transform every local box into the base frame given the printer's
        world/base pose. Returns dicts ready for
        MoveIt2.add_collision_box(id=.., size=.., position=.., quat_xyzw=..):
            [{'id', 'size', 'position', 'quat_xyzw'}, ...]

        Not called yet — this is the drop-in point for collision avoidance. Each
        box's base orientation is the printer orientation composed with the box's
        local rpy; its base position is the printer pose applied to the local
        center."""
        from scipy.spatial.transform import Rotation as R

        R_pb = R.from_quat(list(quat_xyzw))     # printer -> base
        pos = np.asarray(pos, dtype=float)
        result = []
        for i, box in enumerate(self.boxes):
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
