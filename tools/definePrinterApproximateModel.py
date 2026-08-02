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
saving. R / E spin the placed tag +/-90 degrees about its face normal (the
in-plane orientation a click can't choose; re-clicking resets it upright).
The pose is written to the JSON's "markers" list.

The same window also edits the printer's ENTRY ZONE: SHIFT+click a box to add
it to / remove it from the zone (amber wireframe). Zone boxes get
"entry_zone": true in the JSON's box entry. They collide NORMALLY by default —
the zone is the SUBSET of boxes whose collisions a pickup sequence toggles off
while the gripper reaches into the printer (waypoint entry
{'entry_zone_collisions': 'off'}) and back on after withdrawing
({'entry_zone_collisions': 'on'}, printerAutomation.set_entry_zone_collisions;
the global COLLISIONS switch is separate and covers everything). While open
they recolor in Gazebo and their boxes leave the MoveIt planning scene
(visible in RViz); closing the zone restores both. Flag e.g. the door opening
or the bed area the gripper must enter.

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
# boxes in the ENTRY ZONE ("entry_zone": true) — amber, matching the color
# Gazebo shows while a sequence has the zone open (collisions off)
ENTRY_ZONE_BOX_COLOR = [0.95, 0.75, 0.10]


def box_color(i, box):
    """Wireframe color for box i: amber when in the entry zone, else its
    cycle color."""
    return (ENTRY_ZONE_BOX_COLOR if box.get('entry_zone')
            else BOX_COLORS[i % len(BOX_COLORS)])


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

    from scipy.spatial.transform import Rotation as Rot

    geoms = [o3m] if o3m is not None else []
    for i, box in enumerate(model['boxes']):
        color = box_color(i, box)
        c = np.asarray(box['center'], dtype=float)
        s = np.asarray(box['size'], dtype=float)
        # rpy is EXTRINSIC xyz (the SDF/scipy convention the whole pipeline
        # uses) — NOT o3d's get_rotation_matrix_from_xyz, which is intrinsic
        # and silently draws every rotated box wrong
        R = Rot.from_euler('xyz', box.get('rpy', [0, 0, 0])).as_matrix()
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
    normal and the index of the box it belongs to — all in the printer-local
    frame: (face_point, normal, box_index).

    Clicking anywhere near the printer — on the mesh, on a wireframe edge, on a
    box fill — resolves to a flat spot on the collision boxes, which is where a
    marker has to live for the spawned SDF to match what you see here. The
    index is what SHIFT+click entry-zone flagging uses to know which box was
    hit."""
    from scipy.spatial.transform import Rotation as R

    p = np.asarray(point, dtype=float)
    best = None
    for bi, box in enumerate(boxes):
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
            best = (d, face_world, rot.apply(n_local), bi)

    return best[1], best[2], best[3]


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


def write_box_flags(json_path, boxes):
    """Persist each box's entry-zone flag to the JSON, leaving everything else
    untouched. The flag is stored ONLY when True ("entry_zone": true) so
    untouched files keep their exact old shape; un-flagging a box removes the
    key again."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    flagged = 0
    for entry, box in zip(data['boxes'], boxes):
        if box.get('entry_zone'):
            entry['entry_zone'] = True
            flagged += 1
        else:
            entry.pop('entry_zone', None)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    return flagged


def run_marker_placement(printer, model, geoms, json_path, marker_name,
                         marker_size, texture_path, surface_offset):
    """Viewer + CTRL+click marker placement + SHIFT+click entry-zone editing,
    on open3d's GUI renderer.

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
        f"{printer}: CTRL+click places '{marker_name}', "
        f"SHIFT+click edits the entry zone", 1280, 900)

    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(window.renderer)
    widget.scene.set_background([0.10, 0.11, 0.13, 1.0])

    lit = rendering.MaterialRecord()
    lit.shader = 'defaultLit'
    line = rendering.MaterialRecord()
    line.shader = 'unlitLine'
    line.line_width = 2.0

    # boxes are added by BOX index (not geoms index) so a SHIFT+click can
    # re-color exactly the box it toggled; the OBBs in geoms are skipped in
    # favor of these
    def add_box_lineset(bi):
        nm = f"box_{bi}"
        if widget.scene.has_geometry(nm):
            widget.scene.remove_geometry(nm)
        box = boxes[bi]
        c = np.asarray(box['center'], dtype=float)
        s = np.asarray(box['size'], dtype=float)
        # extrinsic-xyz rpy, same convention note as build_geometries
        from scipy.spatial.transform import Rotation as Rot
        Rm = Rot.from_euler('xyz', box.get('rpy', [0, 0, 0])).as_matrix()
        obb = o3d.geometry.OrientedBoundingBox(c, Rm, s)
        ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
        ls.paint_uniform_color(box_color(bi, box))
        widget.scene.add_geometry(nm, ls, line)

    for i, g in enumerate(geoms):
        if isinstance(g, o3d.geometry.OrientedBoundingBox):
            continue
        elif isinstance(g, o3d.geometry.LineSet):
            widget.scene.add_geometry(f"lines_{i}", g, line)
        else:
            widget.scene.add_geometry(f"mesh_{i}", g, lit)
    for bi in range(len(boxes)):
        add_box_lineset(bi)

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

    # ---- side panel: live pose readout + saves
    em = window.theme.font_size
    panel = gui.Vert(0.5 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
    info = gui.Label("CTRL+click the printer to place the marker.\n"
                     "R / E: spin the tag +/-90 deg in its face plane.\n"
                     "Red X = marker normal (points INTO the body).\n"
                     "Blue Z = marker up. Texture shows on the outward face.\n\n"
                     "no marker placed yet")
    panel.add_child(info)
    save_btn = gui.Button(f"Save '{marker_name}' to {os.path.basename(json_path)}")
    save_btn.enabled = False
    panel.add_child(save_btn)

    def flags_summary():
        flagged = [i for i, b in enumerate(boxes) if b.get('entry_zone')]
        return ("SHIFT+click a box to add/remove it from\n"
                "the ENTRY ZONE. Amber wireframe = zone box:\n"
                "a pickup sequence toggles its collisions\n"
                "off while reaching in ({'entry_zone_\n"
                "collisions': 'off'/'on'} waypoints); it\n"
                "collides normally the rest of the time.\n\n"
                + (f"entry-zone boxes: {flagged}" if flagged
                   else "entry zone is empty"))

    flags_info = gui.Label(flags_summary())
    panel.add_child(flags_info)
    flags_btn = gui.Button(f"Save entry zone to {os.path.basename(json_path)}")
    flags_btn.enabled = False
    panel.add_child(flags_btn)
    window.add_child(widget)
    window.add_child(panel)

    def on_layout(ctx):
        r = window.content_rect
        pw = 22 * em
        widget.frame = gui.Rect(r.x, r.y, r.width - pw, r.height)
        panel.frame = gui.Rect(r.get_right() - pw, r.y, pw, r.height)
    window.set_on_layout(on_layout)

    def toggle_box(world_pt):
        _pt, _n, bi = snap_to_box_face(world_pt, boxes)
        flagged = not boxes[bi].get('entry_zone')
        boxes[bi]['entry_zone'] = flagged
        add_box_lineset(bi)
        flags_info.text = flags_summary()
        flags_btn.enabled = True
        print(f"  box {bi}: {'in the ENTRY ZONE' if flagged else 'always-colliding'}")
        window.set_needs_layout()

    def redraw_marker(extra=""):
        """Draw the marker at state's pose and refresh the readout/save."""
        pos, rpy = state['pos'], state['rpy']
        for nm in ('marker_0', 'marker_1', 'marker_2'):
            if widget.scene.has_geometry(nm):
                widget.scene.remove_geometry(nm)
        for j, (geom, mat) in enumerate(
                marker_geometries(pos, rpy, marker_size, texture_path)):
            widget.scene.add_geometry(f"marker_{j}", geom, mat)
        info.text = (f"CTRL+click to move the marker.\n"
                     f"R / E: spin the tag +/-90 deg in its face plane.\n"
                     f"Red X = normal (into body), Blue Z = up.\n"
                     + extra +
                     f"\npos: [{pos[0]:+.4f} {pos[1]:+.4f} {pos[2]:+.4f}]\n"
                     f"rpy: [{rpy[0]:+.4f} {rpy[1]:+.4f} {rpy[2]:+.4f}]")
        save_btn.enabled = True
        window.set_needs_layout()

    def show_marker(world_pt):
        face_pt, normal, _bi = snap_to_box_face(world_pt, boxes)
        pos, rpy = marker_pose_from_face(face_pt, normal, surface_offset)
        state['pos'], state['rpy'] = pos, rpy
        redraw_marker(
            f"\nface normal: [{normal[0]:+.2f} {normal[1]:+.2f} {normal[2]:+.2f}]")
        print(f"  marker '{marker_name}' -> pos={np.round(pos, 4).tolist()} "
              f"rpy={np.round(rpy, 4).tolist()}")

    def rotate_marker(deg):
        """Spin the placed marker about its own face normal (the plate's local
        +X) in 90-degree steps — the one orientation DOF a click can't choose.
        Re-clicking a face resets to the default upright orientation."""
        if state['pos'] is None:
            return
        from scipy.spatial.transform import Rotation as R
        rot = (R.from_euler('xyz', state['rpy'])
               * R.from_euler('x', np.radians(deg)))
        state['rpy'] = rot.as_euler('xyz')
        redraw_marker()
        print(f"  marker '{marker_name}' spun {deg:+.0f} deg -> "
              f"rpy={np.round(state['rpy'], 4).tolist()}")

    def on_key(event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        if event.key == ord('r'):
            rotate_marker(90.0)
            return gui.Widget.EventCallbackResult.HANDLED
        if event.key == ord('e'):
            rotate_marker(-90.0)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    widget.set_on_key(on_key)

    def on_mouse(event):
        if event.type != gui.MouseEvent.Type.BUTTON_DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        if event.is_modifier_down(gui.KeyModifier.CTRL):
            handler = show_marker          # place/move the marker
        elif event.is_modifier_down(gui.KeyModifier.SHIFT):
            handler = toggle_box           # flip the clicked box's collision
        else:
            return gui.Widget.EventCallbackResult.IGNORED

        def depth_callback(depth_image):
            x = event.x - widget.frame.x
            y = event.y - widget.frame.y
            depth = np.asarray(depth_image)[y, x]
            if depth >= 1.0:            # clicked empty space
                return
            world = widget.scene.camera.unproject(
                x, y, depth, widget.frame.width, widget.frame.height)
            app.post_to_main_thread(window, lambda: handler(world))

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

    def on_save_flags():
        flagged = write_box_flags(json_path, boxes)
        flags_info.text = (flags_summary()
                           + f"\n\nsaved: {flagged} entry-zone box(es)")
        print(f"  wrote entry-zone flags ({flagged} boxes) -> {json_path}")
        window.set_needs_layout()
    flags_btn.set_on_clicked(on_save_flags)

    app.run()


def main():
    # ================= CONFIG (edit these; no command-line args) =============
    # any model in models/printers/ works here: 'a1', 'a1_mini',
    # 'scrape_fixture', ... — the JSON's source_mesh is drawn underneath when
    # it's readable (.stp needs trimesh+cascadio; without them the viewer
    # still opens, just with the boxes and no mesh underlay)
    PRINTER = 'a1'
    SOLID_BOXES = False     # True adds translucent-colored box fills
    SCREENSHOT = None       # set to a PNG path to render headlessly; None = window
    CHECK_ONLY = False      # True = build geometries + print stats, no window

    PLACE_MARKER = True     # True = interactive editor window: CTRL+click
    # places the marker, SHIFT+click adds/removes the clicked box from the
    # ENTRY ZONE ("entry_zone": true in the JSON): colliding by default, but a
    # pickup sequence toggles the zone's collisions off/on mid-run
    # ({'entry_zone_collisions': ...} waypoint entries) — recolored in Gazebo,
    # removed/re-added in RViz
    MARKER_NAME = 'door'    # entry in the JSON's "markers"; re-placing replaces it
    MARKER_SIZE = 0.025      # marker edge length (m); stored per mount in the JSON
    MARKER_TEXTURE = 'materials/textures/marker6x6_0.png'   # preview art only:
    # which ID a spawned printer actually wears is a spawn-time choice, since the
    # same model can be spawned several times with different markers.
    SURFACE_OFFSET = 0.0001  # push the plate this far out of the face (m)
    # ========================================================================

    import open3d as o3d
    interactive = PLACE_MARKER and not SCREENSHOT and not CHECK_ONLY
    # the interactive editor recolors box wireframes as they're toggled; solid
    # fills can't be updated in place, so they're dropped there
    model, geoms = build_geometries(
        PRINTER, solid_boxes=SOLID_BOXES and not interactive)

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

    if interactive:
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
