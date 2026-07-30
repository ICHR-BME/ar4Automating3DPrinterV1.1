#!/usr/bin/env python3
"""Interactive 3D overlay of a printer's real mesh vs. its primitive-box model.

Loads the source STEP/STL mesh (solid gray) and the boxes from
models/printers/<name>.json (colored wireframe), placed in the SAME recentered
frame the boxes were generated in, and opens an orbitable open3d window so you
can verify the boxes envelop the printer before trusting them for collision.

PLACE_MARKER mode additionally lets you put an ArUco marker on the printer by
CTRL+clicking it. The click is snapped to the nearest collision-box face, so the
marker lands flat on the box model even if you clicked the mesh, and it is drawn
where it would actually sit: texture on the outward face only, plain gray
backing, plus an axis triad so you can confirm it faces the right way before
saving. The pose is written to the JSON's "markers" list.

This works because the box frame IS the printer-local frame (origin at the mesh
AABB center, +Z up) — the same frame Simulated3DPrinter places markers in. A
marker measured on the real robot can't give this: teachMarkersByHand.py finds
the marker in base frame, and without the printer's own pose there is nothing to
subtract.

Configure the run by editing the variables in the CONFIG block in main() below
(no command-line arguments). Reads units_scale / rotate_z_deg from the JSON so
the mesh matches the boxes exactly. STL loads through open3d directly; STEP
still needs the tool's loader (trimesh + cascadio).
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


def load_source_mesh(mesh_path, units_scale, rotate_z_deg):
    """The source mesh in the box frame: scaled to meters, optional yaw, then
    recentered on its AABB — the exact transform printer_mesh_to_boxes.py used,
    so mesh and boxes overlay.

    STL/OBJ/PLY go through open3d's own reader (no trimesh needed, which matters
    because trimesh/cascadio aren't always installed); STEP falls back to the
    tool's loader. Returns None if the mesh can't be read, so the viewer still
    opens with just the boxes."""
    import open3d as o3d

    ext = os.path.splitext(mesh_path)[1].lower()
    if not os.path.isfile(mesh_path):
        print(f"  ! source mesh not found: {mesh_path} — showing boxes only")
        return None

    if ext in ('.stl', '.obj', '.ply', '.glb'):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not mesh.has_triangles():
            print(f"  ! open3d read no triangles from {mesh_path} — boxes only")
            return None
        verts = np.asarray(mesh.vertices, dtype=float) * float(units_scale)
    else:
        try:
            tri, _ = load_mesh(mesh_path, units_scale, rotate_z_deg)
        except ImportError as e:
            print(f"  ! {ext} needs trimesh/cascadio ({e}) — showing boxes only")
            return None
        # load_mesh already scaled/rotated/recentered
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(tri.vertices, dtype=float)),
            o3d.utility.Vector3iVector(np.asarray(tri.faces, dtype=np.int32)))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.62, 0.62, 0.66])
        return mesh

    if rotate_z_deg:
        a = np.radians(rotate_z_deg)
        Rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                       [np.sin(a),  np.cos(a), 0.0],
                       [0.0,        0.0,       1.0]])
        verts = verts @ Rz.T
    verts -= (verts.min(axis=0) + verts.max(axis=0)) / 2.0

    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.62, 0.62, 0.66])
    return mesh


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
    o3m = load_source_mesh(mesh_path, model.get('units_scale', 0.001),
                           model.get('rotate_z_deg', 0.0))

    geoms = [o3m] if o3m is not None else []
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


# ---------------------------------------------------------------- marker place

def snap_to_box_face(point, boxes):
    """Nearest point on the surface of the box model, with that face's outward
    normal. Both are returned in the printer-local frame.

    Clicking anywhere near the printer — on the mesh, on a wireframe edge, on a
    box fill — resolves to a flat spot on the collision boxes, which is where a
    marker has to live for the spawned SDF to match what you see here."""
    from scipy.spatial.transform import Rotation as R

    p = np.asarray(point, dtype=float)
    best = None
    for box in boxes:
        c = np.asarray(box['center'], dtype=float)
        half = np.asarray(box['size'], dtype=float) / 2.0
        rot = R.from_euler('xyz', box.get('rpy', [0.0, 0.0, 0.0]))

        local = rot.inv().apply(p - c)
        clamped = np.clip(local, -half, half)
        outside = local - clamped

        if np.linalg.norm(outside) > 1e-9:
            # point is off the box: the surface point is the clamped one, and the
            # face is the axis it sticks out furthest along
            axis = int(np.argmax(np.abs(outside)))
            sign = 1.0 if outside[axis] >= 0 else -1.0
            face_local = clamped
        else:
            # point is inside: exit through the nearest face plane
            axis = int(np.argmin(half - np.abs(local)))
            sign = 1.0 if local[axis] >= 0 else -1.0
            face_local = local.copy()
            face_local[axis] = sign * half[axis]

        n_local = np.zeros(3)
        n_local[axis] = sign

        face_world = rot.apply(face_local) + c
        d = float(np.linalg.norm(face_world - p))
        if best is None or d < best[0]:
            best = (d, face_world, rot.apply(n_local))

    return best[1], best[2]


def marker_pose_from_face(point, normal, surface_offset):
    """Marker pose (pos, rpy) in the printer-local frame for a point on a face.

    Orientation matches the convention already in simulated3DPrinter.py: a
    marker on the front (-Y) face is placed with a plain yaw of +90 deg, which
    puts the plate's local +X along printer +Y — i.e. +X is the INWARD normal —
    and leaves its +Z along printer +Z. Reproducing that here means a marker
    placed by clicking renders exactly like the existing door/static markers.

    'Up' is printer +Z projected into the marker plane, so the tag stays upright
    on any vertical face; on a horizontal face that projection is degenerate, so
    +Y is used and up points toward the printer's back.

    rpy is extrinsic xyz, matching SDF <pose>.
    """
    from scipy.spatial.transform import Rotation as R

    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)

    up_ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(n, up_ref))) > 0.99:
        up_ref = np.array([0.0, 1.0, 0.0])
    up = up_ref - np.dot(up_ref, n) * n
    up = up / np.linalg.norm(up)

    x = -n                       # plate +X points into the body (see docstring)
    y = np.cross(up, x)
    rot = R.from_matrix(np.column_stack([x, y, up]))

    pos = np.asarray(point, dtype=float) + n * surface_offset
    return pos, rot.as_euler('xyz')


def marker_geometries(pos, rpy, size, texture_path):
    """The marker as it will sit on the printer: a textured quad on the OUTWARD
    face, a plain dark backing behind it, and an axis triad.

    Only the outward face carries the texture, so a marker that ends up facing
    into the printer is obvious at a glance — from behind you see flat gray.

    Returns [(geometry, material_or_None), ...] ready for an Open3DScene.
    """
    import open3d as o3d
    from open3d.visualization import rendering
    from scipy.spatial.transform import Rotation as R

    rot = R.from_euler('xyz', rpy).as_matrix()
    pos = np.asarray(pos, dtype=float)
    h = size / 2.0

    def place(geom):
        geom.rotate(rot, center=(0, 0, 0))
        geom.translate(pos)
        return geom

    # textured quad, sitting just proud of the backing on the -X (outward) side.
    # Local frame: +X inward, +Y right, +Z up -> the quad spans Y and Z.
    quad = o3d.geometry.TriangleMesh()
    quad.vertices = o3d.utility.Vector3dVector(np.array([
        [-0.0021, -h, -h], [-0.0021, h, -h], [-0.0021, h, h], [-0.0021, -h, h]]))
    quad.triangles = o3d.utility.Vector3iVector(np.array([[0, 2, 1], [0, 3, 2]]))
    quad.triangle_uvs = o3d.utility.Vector2dVector(np.array([
        [0.0, 0.0], [1.0, 1.0], [1.0, 0.0],
        [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    quad.compute_vertex_normals()

    quad_mat = rendering.MaterialRecord()
    quad_mat.shader = 'defaultUnlit'          # unlit: the tag reads at any angle
    if texture_path and os.path.isfile(texture_path):
        img = o3d.io.read_image(texture_path)
        quad.textures = [img]
        quad.triangle_material_ids = o3d.utility.IntVector([0, 0])
        quad_mat.albedo_img = img
    else:
        print(f"  ! marker texture not found: {texture_path} (drawing it plain)")
        quad.paint_uniform_color([0.95, 0.95, 0.95])

    # backing plate: opaque, clearly not the tag, so the back face is unmistakable
    back = o3d.geometry.TriangleMesh.create_box(0.002, size, size)
    back.translate([-0.002, -h, -h])
    back.compute_vertex_normals()
    back.paint_uniform_color([0.15, 0.15, 0.18])
    back_mat = rendering.MaterialRecord()
    back_mat.shader = 'defaultLit'

    triad = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size * 0.9)
    triad_mat = rendering.MaterialRecord()
    triad_mat.shader = 'defaultLit'

    return [(place(quad), quad_mat), (place(back), back_mat),
            (place(triad), triad_mat)]


def write_marker(json_path, name, size, pos, rpy):
    """Insert/replace the named entry in the JSON's "markers" list, leaving the
    rest of the file (boxes, footprint, provenance) untouched."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    entry = {
        'name': name,
        'size': round(float(size), 6),
        'pos': [round(float(v), 6) for v in pos],
        'rpy': [round(float(v), 6) for v in rpy],
    }
    markers = [m for m in data.get('markers', []) if m.get('name') != name]
    markers.append(entry)
    data['markers'] = markers

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    return entry


def run_marker_placement(printer, model, geoms, json_path, marker_name,
                         marker_size, texture_path, surface_offset):
    """Viewer + CTRL+click marker placement, on open3d's GUI renderer.

    Picking is depth-buffer based: the clicked pixel's depth is unprojected back
    into the scene, which works against the solid mesh and box fills alike (the
    legacy point-picking visualizer only picks point clouds, which would have
    meant scattering visible sample points over the model just to click them)."""
    import open3d as o3d
    from open3d.visualization import gui, rendering

    boxes = model['boxes']
    state = {'pos': None, 'rpy': None}

    app = gui.Application.instance
    app.initialize()
    window = app.create_window(
        f"{printer}: CTRL+click to place '{marker_name}'", 1280, 900)

    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(window.renderer)
    widget.scene.set_background([0.10, 0.11, 0.13, 1.0])

    lit = rendering.MaterialRecord()
    lit.shader = 'defaultLit'
    line = rendering.MaterialRecord()
    line.shader = 'unlitLine'
    line.line_width = 2.0

    for i, g in enumerate(geoms):
        if isinstance(g, o3d.geometry.OrientedBoundingBox):
            ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(g)
            ls.paint_uniform_color(g.color)
            widget.scene.add_geometry(f"box_{i}", ls, line)
        elif isinstance(g, o3d.geometry.LineSet):
            widget.scene.add_geometry(f"lines_{i}", g, line)
        else:
            widget.scene.add_geometry(f"mesh_{i}", g, lit)

    # markers already saved in the JSON, so re-running shows what you placed
    # last time and a CTRL+click just moves it
    for m in model.get('markers', []):
        if m.get('name') == marker_name:
            state['pos'], state['rpy'] = m['pos'], m['rpy']
            for j, (geom, mat) in enumerate(marker_geometries(
                    m['pos'], m['rpy'], m.get('size', marker_size), texture_path)):
                widget.scene.add_geometry(f"marker_{j}", geom, mat)

    bounds = widget.scene.bounding_box
    widget.setup_camera(60.0, bounds, bounds.get_center())

    # ---- side panel: live pose readout + save
    em = window.theme.font_size
    panel = gui.Vert(0.5 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
    info = gui.Label("CTRL+click the printer to place the marker.\n"
                     "Red X = marker normal (points INTO the body).\n"
                     "Blue Z = marker up. Texture shows on the outward face.\n\n"
                     "no marker placed yet")
    panel.add_child(info)
    save_btn = gui.Button(f"Save '{marker_name}' to {os.path.basename(json_path)}")
    save_btn.enabled = False
    panel.add_child(save_btn)
    window.add_child(widget)
    window.add_child(panel)

    def on_layout(ctx):
        r = window.content_rect
        pw = 22 * em
        widget.frame = gui.Rect(r.x, r.y, r.width - pw, r.height)
        panel.frame = gui.Rect(r.get_right() - pw, r.y, pw, r.height)
    window.set_on_layout(on_layout)

    def show_marker(world_pt):
        face_pt, normal = snap_to_box_face(world_pt, boxes)
        pos, rpy = marker_pose_from_face(face_pt, normal, surface_offset)
        state['pos'], state['rpy'] = pos, rpy

        for nm in ('marker_0', 'marker_1', 'marker_2'):
            if widget.scene.has_geometry(nm):
                widget.scene.remove_geometry(nm)
        for j, (geom, mat) in enumerate(
                marker_geometries(pos, rpy, marker_size, texture_path)):
            widget.scene.add_geometry(f"marker_{j}", geom, mat)

        info.text = (f"CTRL+click to move the marker.\n"
                     f"Red X = normal (into body), Blue Z = up.\n\n"
                     f"face normal: [{normal[0]:+.2f} {normal[1]:+.2f} {normal[2]:+.2f}]\n"
                     f"pos: [{pos[0]:+.4f} {pos[1]:+.4f} {pos[2]:+.4f}]\n"
                     f"rpy: [{rpy[0]:+.4f} {rpy[1]:+.4f} {rpy[2]:+.4f}]")
        save_btn.enabled = True
        print(f"  marker '{marker_name}' -> pos={np.round(pos, 4).tolist()} "
              f"rpy={np.round(rpy, 4).tolist()}")
        window.set_needs_layout()

    def on_mouse(event):
        if not (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                and event.is_modifier_down(gui.KeyModifier.CTRL)):
            return gui.Widget.EventCallbackResult.IGNORED

        def depth_callback(depth_image):
            x = event.x - widget.frame.x
            y = event.y - widget.frame.y
            depth = np.asarray(depth_image)[y, x]
            if depth >= 1.0:            # clicked empty space
                return
            world = widget.scene.camera.unproject(
                x, y, depth, widget.frame.width, widget.frame.height)
            app.post_to_main_thread(window, lambda: show_marker(world))

        widget.scene.scene.render_to_depth_image(depth_callback)
        return gui.Widget.EventCallbackResult.HANDLED

    widget.set_on_mouse(on_mouse)

    def on_save():
        if state['pos'] is None:
            return
        entry = write_marker(json_path, marker_name, marker_size,
                             state['pos'], state['rpy'])
        info.text = f"saved to {os.path.basename(json_path)}:\n{entry}"
        print(f"  wrote markers['{marker_name}'] -> {json_path}")
        window.set_needs_layout()
    save_btn.set_on_clicked(on_save)

    app.run()


def main():
    # ================= CONFIG (edit these; no command-line args) =============
    PRINTER = 'a1_mini'          # which model to view: 'a1' or 'a1_mini'
    # (a1_mini's source is a .stp, which needs trimesh+cascadio; without them
    # the viewer still opens, just with the boxes and no mesh underlay)
    SOLID_BOXES = False     # True adds translucent-colored box fills
    SCREENSHOT = None       # set to a PNG path to render headlessly; None = window
    CHECK_ONLY = False      # True = build geometries + print stats, no window

    PLACE_MARKER = True     # True = CTRL+click places a marker (needs a window)
    MARKER_NAME = 'door'    # entry in the JSON's "markers"; re-placing replaces it
    MARKER_SIZE = 0.025      # marker edge length (m); stored per mount in the JSON
    MARKER_TEXTURE = 'materials/textures/marker6x6_0.png'   # preview art only:
    # which ID a spawned printer actually wears is a spawn-time choice, since the
    # same model can be spawned several times with different markers.
    SURFACE_OFFSET = 0.0001  # push the plate this far out of the face (m)
    # ========================================================================

    import open3d as o3d
    model, geoms = build_geometries(PRINTER, solid_boxes=SOLID_BOXES)

    cov = model.get('coverage', {})
    print(f"[{PRINTER}] {cov.get('num_boxes','?')} boxes | "
          f"coverage={cov.get('voxel_coverage','?')} | "
          f"max_protrusion={cov.get('max_protrusion_mm','?')}mm | "
          f"footprint(m)={ {k: round(v,3) for k,v in model['footprint'].items()} }")
    for m in model.get('markers', []):
        print(f"  existing marker '{m.get('name')}': pos={m['pos']} rpy={m['rpy']}")

    if CHECK_ONLY:
        print(f"built {len(geoms)} geometries (mesh + {len(model['boxes'])} boxes + axes) OK")
        return 0

    if PLACE_MARKER and not SCREENSHOT:
        json_path = os.path.join(PRINTERS_DIR, f'{PRINTER}.json')
        texture_path = os.path.join(
            REPO_ROOT, 'models', 'aruco_marker', MARKER_TEXTURE)
        run_marker_placement(PRINTER, model, geoms, json_path, MARKER_NAME,
                             MARKER_SIZE, texture_path, SURFACE_OFFSET)
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
