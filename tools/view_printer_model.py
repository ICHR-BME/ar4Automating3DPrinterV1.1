#!/usr/bin/env python3
"""Interactive 3D overlay of a printer's real mesh vs. its primitive-box model.

Loads the source STEP/STL mesh (solid gray) and the boxes from
models/printers/<name>.json (colored wireframe), placed in the SAME recentered
frame the boxes were generated in, and opens an orbitable open3d window so you
can verify the boxes envelop the printer before trusting them for collision.

Configure the run by editing the variables in the CONFIG block in main() below
(no command-line arguments). Reads units_scale / rotate_z_deg from the JSON so
the mesh matches the boxes exactly. Requires open3d + the tool's mesh loader
(trimesh/cascadio for STEP).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from printer_mesh_to_boxes import load_mesh, PRINTERS_DIR, REPO_ROOT  # noqa: E402

# distinct colors cycled across boxes so the decomposition is legible
BOX_COLORS = [
    [0.90, 0.10, 0.10], [0.10, 0.55, 0.90], [0.10, 0.75, 0.20],
    [0.95, 0.60, 0.05], [0.70, 0.20, 0.85], [0.05, 0.75, 0.75],
    [0.85, 0.35, 0.55], [0.55, 0.55, 0.10],
]


def build_geometries(name, solid_boxes=False):
    """Overlay the full source mesh (solid gray) with the collision-box
    wireframes, so you can compare the real model against the boxes. Set
    solid_boxes=True to also add translucent box fills."""
    import open3d as o3d

    json_path = os.path.join(PRINTERS_DIR, f'{name}.json')
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"{json_path} not found — generate it first with "
                                f"tools/printer_mesh_to_boxes.py")
    with open(json_path) as f:
        model = json.load(f)

    mesh_path = model['source_mesh']
    if not os.path.isabs(mesh_path):
        mesh_path = os.path.join(REPO_ROOT, mesh_path)
    # same transform used to build the boxes -> mesh lands in the box frame
    tri, verts = load_mesh(mesh_path, model.get('units_scale', 0.001),
                           model.get('rotate_z_deg', 0.0))

    o3m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(tri.vertices, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray(tri.faces, dtype=np.int32)))
    o3m.compute_vertex_normals()
    o3m.paint_uniform_color([0.62, 0.62, 0.66])

    geoms = [o3m]
    for i, box in enumerate(model['boxes']):
        color = BOX_COLORS[i % len(BOX_COLORS)]
        c = np.asarray(box['center'], dtype=float)
        s = np.asarray(box['size'], dtype=float)
        R = o3d.geometry.get_rotation_matrix_from_xyz(box.get('rpy', [0, 0, 0]))
        obb = o3d.geometry.OrientedBoundingBox(c, R, s)
        obb.color = color
        geoms.append(obb)
        if solid_boxes:
            fill = o3d.geometry.TriangleMesh.create_box(*[max(v, 1e-4) for v in s])
            fill.translate(-s / 2.0)
            fill.rotate(R, center=(0, 0, 0))
            fill.translate(c)
            fill.compute_vertex_normals()
            fill.paint_uniform_color(color)
            geoms.append(fill)

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.08, origin=[0, 0, 0]))
    return model, geoms


def main():
    # ================= CONFIG (edit these; no command-line args) =============
    PRINTER = 'a1_mini'          # which model to view: 'a1' or 'a1_mini'
    SOLID_BOXES = False     # True adds translucent-colored box fills
    SCREENSHOT = None       # set to a PNG path to render headlessly; None = window
    CHECK_ONLY = False      # True = build geometries + print stats, no window
    # ========================================================================

    import open3d as o3d
    model, geoms = build_geometries(PRINTER, solid_boxes=SOLID_BOXES)

    cov = model.get('coverage', {})
    print(f"[{PRINTER}] {cov.get('num_boxes','?')} boxes | "
          f"coverage={cov.get('voxel_coverage','?')} | "
          f"max_protrusion={cov.get('max_protrusion_mm','?')}mm | "
          f"footprint(m)={ {k: round(v,3) for k,v in model['footprint'].items()} }")

    if CHECK_ONLY:
        print(f"built {len(geoms)} geometries (mesh + {len(model['boxes'])} boxes + axes) OK")
        return 0

    title = f"{PRINTER}: mesh (gray) vs {len(model['boxes'])} collision boxes"
    if SCREENSHOT:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=960)
        for g in geoms:
            vis.add_geometry(g)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(SCREENSHOT, do_render=True)
        vis.destroy_window()
        print(f"wrote {SCREENSHOT}")
    else:
        o3d.visualization.draw_geometries(geoms, window_name=title,
                                          mesh_show_back_face=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
