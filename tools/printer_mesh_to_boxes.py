#!/usr/bin/env python3
"""Approximate a 3D-printer mesh with a few primitive boxes for Gazebo + MoveIt.

Reads a real printer mesh (STL/OBJ/PLY/GLB — NOT STEP), slices it into a handful
of horizontal (Z) bands, and bounds each band with an axis-aligned box in the
printer-local frame (origin at the mesh's geometric center, the same frame the
Gazebo spawner and, later, MoveIt collision boxes use). Adjacent bands with
near-identical XY extents are merged so the result is a *few* boxes, not one per
slice.

Why Z-band bounding (vs. convex decomposition): for collision avoidance a box
that fully *contains* the printer is the safe error direction — the arm keeps
the plate clear even where the approximation bulges into empty space. So voxel
coverage is ~100% by construction; the number worth watching is the
over-approximation ratio (how much empty space the boxes add), which drops as
you add bands.

Output: models/printers/<name>.json (box list + footprint + coverage report) and
models/printers/<name>_preview.png (mesh points + box overlay to eyeball).

Requires: trimesh + cascadio (STEP/STL/OBJ/PLY/GLB loading; multi-body assemblies
are merged), numpy, matplotlib. Install standalone:
    python3 -m pip install --user --break-system-packages trimesh cascadio
STEP/STP are supported via cascadio (OpenCASCADE tessellation). PRT (Creo) and
SLDPRT (SolidWorks) are proprietary and NOT supported — export/convert to STEP
or STL first.

Configure the run by editing the CONFIG block in main() (no command-line args):
choose the printer, mesh path, and the voxel tightness knobs, then:
    python3 tools/printer_mesh_to_boxes.py
"""

import json
import os
import sys

import numpy as np

# repo root = one level above tools/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRINTERS_DIR = os.path.join(REPO_ROOT, 'models', 'printers')
MESHES_DIR = os.path.join(PRINTERS_DIR, 'meshes')

# Official Bambu Lab published outer dimensions (W x D x H, meters) used only as
# a sanity hint against the loaded mesh's bounding box — NOT to build the model.
KNOWN_DIMS = {
    'a1':      {'footprint': (0.385, 0.410, 0.458), 'bed': (0.256, 0.256)},
    'a1_mini': {'footprint': (0.347, 0.315, 0.365), 'bed': (0.180, 0.180)},
}

DOWNLOAD_HINT = """\
No mesh found. Download a STEP/STP or STL/OBJ/PLY (NOT PRT/SLDPRT) and save it as:
  {path}
(or pass --mesh <path> with a .step extension)
Sources (GrabCAD preferred; needs a free login — grab the STEP file):
  A1 mini : https://grabcad.com/library/bambu-lab-a1-mini-2/files
  A1      : https://grabcad.com/library/bambu-lab-a1-combo-3d-printer-ams-module-1
  STL alt : https://www.printables.com/model/590682-bambu-labs-a1-mini
"""


def load_mesh(path, scale, rotate_z_deg):
    """Load a mesh (STEP via cascadio->OBJ; STL/OBJ/PLY/GLB natively), merging any
    multi-body assembly into a single mesh. Scale to meters, optional yaw so the
    operator side faces -Y, then recenter so the AABB center is at the origin.

    STEP is tessellated with cascadio.step_to_obj (OBJ, not GLB: trimesh's GLB
    reader trips over cascadio's material-less geometry). cascadio's OBJ keeps
    native STEP units (mm), so the same default --scale 0.001 works as for STLs."""
    import trimesh
    import tempfile

    ext = os.path.splitext(path)[1].lower()
    if ext in ('.step', '.stp'):
        import cascadio
        tmp_obj = tempfile.NamedTemporaryFile(suffix='.obj', delete=False).name
        try:
            cascadio.step_to_obj(path, tmp_obj)
            mesh = trimesh.load(tmp_obj, force='mesh')
        finally:
            if os.path.exists(tmp_obj):
                os.unlink(tmp_obj)
    else:
        # force='mesh' concatenates all bodies in a scene into one Trimesh
        mesh = trimesh.load(path, force='mesh')

    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        raise ValueError(f"could not read a mesh from {path} "
                         f"(PRT/SLDPRT are unsupported — export to STEP or STL)")

    verts = np.asarray(mesh.vertices, dtype=float) * float(scale)

    if rotate_z_deg:
        a = np.radians(rotate_z_deg)
        Rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                       [np.sin(a),  np.cos(a), 0.0],
                       [0.0,        0.0,       1.0]])
        verts = verts @ Rz.T

    # recenter: AABB center -> origin (matches the sim printer-local frame, where
    # the printer center sits at self.pos and walls straddle it)
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    verts -= center

    mesh.vertices = verts
    return mesh, verts


def sample_points(mesh, n):
    """Dense surface point sample (in the mesh's current, recentered frame).
    Falls back to raw vertices if surface sampling is unavailable."""
    import trimesh
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, n)
        return np.asarray(pts, dtype=float)
    except Exception:
        return np.asarray(mesh.vertices, dtype=float)


def _grow_box(occ, seed, min_fill):
    """Grow a maximal axis-aligned box of occupied voxels around `seed`.
    Expands one face-layer at a time in each of the 6 directions while the new
    layer is at least `min_fill` occupied (1.0 = fully solid; <1 tolerates small
    holes, giving fewer/larger boxes). Returns (lo_idx, hi_idx) inclusive."""
    lo = np.array(seed); hi = np.array(seed)
    nx, ny, nz = occ.shape
    improved = True
    while improved:
        improved = False
        for axis in range(3):
            # expand +axis
            if hi[axis] + 1 < occ.shape[axis]:
                sl = [slice(lo[0], hi[0] + 1), slice(lo[1], hi[1] + 1), slice(lo[2], hi[2] + 1)]
                sl[axis] = hi[axis] + 1
                if occ[tuple(sl)].mean() >= min_fill:
                    hi[axis] += 1; improved = True
            # expand -axis
            if lo[axis] - 1 >= 0:
                sl = [slice(lo[0], hi[0] + 1), slice(lo[1], hi[1] + 1), slice(lo[2], hi[2] + 1)]
                sl[axis] = lo[axis] - 1
                if occ[tuple(sl)].mean() >= min_fill:
                    lo[axis] -= 1; improved = True
    return lo, hi


def decompose_boxes(pts, voxel_mm=20.0, dilate=1, min_fill=1.0, min_box_voxels=2,
                    shrink_margin_mm=3.0):
    """Cover the printer's MATERIAL with a small set of axis-aligned boxes,
    leaving hollow/empty space uncovered (the fix for Z-band's over-inclusion).

    Voxelize the surface sample, optionally dilate for a conservative margin and
    to solidify thin shells, then greedily grow maximal occupied boxes until all
    material is covered. Boxes may overlap (fine for collision). Returns
    (boxes, stats) where stats reports the empty-space penalty."""
    from scipy import ndimage

    v = voxel_mm / 1000.0
    lo_world = pts.min(axis=0)
    idx = np.floor((pts - lo_world) / v).astype(int)
    dims = idx.max(axis=0) + 1
    occ = np.zeros(tuple(dims), dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    material = occ.copy()                       # undilated = true material proxy
    if dilate > 0:
        occ = ndimage.binary_dilation(occ, iterations=int(dilate))

    remaining = occ.copy()
    vox_boxes = []                              # (lo, hi) index boxes
    while remaining.any():
        seed = np.argwhere(remaining)[0]
        lo, hi = _grow_box(occ, seed, min_fill)
        remaining[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = False
        if int(np.prod(hi - lo + 1)) < min_box_voxels:
            continue                            # drop negligible single-voxel specks
        vox_boxes.append((lo, hi))

    # drop boxes fully contained in another (greedy overlap can produce these)
    keep = []
    for a in sorted(vox_boxes, key=lambda b: -int(np.prod(b[1] - b[0] + 1))):
        if not any(np.all(a[0] >= k[0]) and np.all(a[1] <= k[1]) for k in keep):
            keep.append(a)
    vox_boxes = keep

    # SHRINK-TO-FIT: each voxel box carries ~1 voxel of half-empty boundary plus
    # the dilation margin. Clamp every box to the ACTUAL mesh points inside it
    # (+ a small safety margin), stripping that void without losing material.
    # Boxes that end up empty (pure dilation artifacts) are dropped.
    margin = shrink_margin_mm / 1000.0
    shape = np.array(occ.shape)
    covered = np.zeros_like(occ)
    boxes = []
    for lo, hi in vox_boxes:
        c_lo = lo_world + lo * v
        c_hi = lo_world + (hi + 1) * v
        inside = np.all((pts >= c_lo) & (pts <= c_hi), axis=1)
        if inside.sum() == 0:
            continue
        q = pts[inside]
        b_lo = q.min(axis=0) - margin
        b_hi = q.max(axis=0) + margin
        boxes.append({
            'center': [float(x) for x in (b_lo + b_hi) / 2.0],
            'size': [float(x) for x in (b_hi - b_lo)],
            'rpy': [0.0, 0.0, 0.0],
        })
        # rasterize the shrunk box for the stats grid
        i_lo = np.clip(np.floor((b_lo - lo_world) / v).astype(int), 0, shape - 1)
        i_hi = np.clip(np.ceil((b_hi - lo_world) / v).astype(int), 0, shape - 1)
        covered[i_lo[0]:i_hi[0] + 1, i_lo[1]:i_hi[1] + 1, i_lo[2]:i_hi[2] + 1] = True

    # honest empty-space metric: of the volume the boxes enclose, the fraction
    # that falls OUTSIDE the (dilated) material — i.e. genuine over-approximation
    # from min_fill<1 bridging gaps. The undilated shell would undercount solids.
    boxed_vox = int(covered.sum())
    over_vox = int((covered & ~occ).sum())
    aabb = (pts.max(axis=0) - pts.min(axis=0))
    stats = {
        'num_boxes': len(boxes),
        'voxel_mm': voxel_mm,
        'dilate': dilate,
        'min_fill': min_fill,
        # over-approximation: boxed volume outside the material (lower = tighter)
        'empty_fraction': round(over_vox / boxed_vox, 3) if boxed_vox else 0.0,
        # how much of the bounding box the boxes fill (Z-band filled ~all of it;
        # excluding the central void drops this well below 1)
        'fill_vs_aabb': round(boxed_vox * v ** 3 / float(np.prod(aabb)), 3),
        'material_volume_m3': round(int(material.sum()) * v ** 3, 6),
        'boxed_volume_m3': round(boxed_vox * v ** 3, 6),
    }
    return boxes, stats


def _remesh_clean(pts, remesh_mm, dilate):
    """Voxelize the surface sample and marching-cubes it into ONE clean
    watertight manifold — repairing the raw STEP tessellation (thousands of
    non-manifold components) so CoACD won't crash. `dilate` grows the occupancy
    outward first so the remesh CONTAINS the original surface (conservative)."""
    from scipy import ndimage
    from skimage import measure
    import trimesh

    p = remesh_mm / 1000.0
    lo = pts.min(axis=0)
    idx = np.floor((pts - lo) / p).astype(int)
    occ = np.zeros(idx.max(axis=0) + 2, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if dilate > 0:
        occ = ndimage.binary_dilation(occ, iterations=int(dilate))
    occ = ndimage.binary_fill_holes(occ)     # closes enclosed cavities, not the open bay
    occ = np.pad(occ, 1)
    mv, mf, _, _ = measure.marching_cubes(occ.astype(float), level=0.5, spacing=(p, p, p))
    mv += lo - p                              # undo pad + shift to the metric frame
    m = trimesh.Trimesh(mv, mf)
    m.fix_normals()
    return m


def decompose_coacd(pts, remesh_mm=6.0, dilate=1, threshold=0.1, max_hulls=100,
                    shrink_margin_mm=3.0):
    """Approximate-convex-decomposition box cover (CoACD). Remesh -> CoACD convex
    pieces -> axis-aligned box per piece, then shrink each box to the real mesh
    points inside it. Fewer, tighter boxes than the voxel greedy cover, still
    plain AABBs (cheap collision, easy to crop). Returns (boxes, stats).
    Needs: coacd, scikit-image (pip install --user --break-system-packages ...)."""
    import coacd
    import trimesh
    coacd.set_log_level("error")

    clean = _remesh_clean(pts, remesh_mm, dilate)
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(clean.vertices), np.asarray(clean.faces)),
        threshold=threshold, preprocess_mode="auto", max_convex_hull=max_hulls,
        merge=True)

    margin = shrink_margin_mm / 1000.0
    boxes = []
    for v, f in parts:
        b = trimesh.Trimesh(v, f).bounds        # AABB of the (conservative) piece
        blo, bhi = np.array(b[0]), np.array(b[1])
        inside = np.all((pts >= blo) & (pts <= bhi), axis=1)
        if inside.sum() == 0:
            continue                            # piece was pure dilation margin
        q = pts[inside]                          # shrink to real material + margin
        blo = q.min(axis=0) - margin
        bhi = q.max(axis=0) + margin
        boxes.append({
            'center': [float(x) for x in (blo + bhi) / 2.0],
            'size': [float(x) for x in (bhi - blo)],
            'rpy': [0.0, 0.0, 0.0],
        })

    stats = {'num_boxes': len(boxes), 'method': 'coacd', 'threshold': threshold,
             'remesh_mm': remesh_mm, 'dilate': dilate}
    return boxes, stats


def _crop_bounds(crop_remove):
    """[(cx,cy,cz,sx,sy,sz), ...] -> [(lo, hi), ...] as numpy arrays."""
    out = []
    for cx, cy, cz, sx, sy, sz in crop_remove:
        c = np.array([cx, cy, cz], float); h = np.array([sx, sy, sz], float) / 2.0
        out.append((c - h, c + h))
    return out


def _box_minus(blo, bhi, clo, chi):
    """Axis-aligned box difference: box \\ crop as up to 6 non-overlapping boxes
    (standard slab carve). Returns [(lo,hi), ...]; the box unchanged if no real
    overlap. This is what actually guarantees no box remains inside the crop."""
    ilo = np.maximum(blo, clo); ihi = np.minimum(bhi, chi)
    if np.any(ilo >= ihi):
        return [(blo, bhi)]                       # no volumetric overlap
    out = []
    lo = blo.astype(float).copy(); hi = bhi.astype(float).copy()
    for ax in range(3):
        if lo[ax] < ilo[ax] - 1e-9:               # slab below the crop on this axis
            nlo = lo.copy(); nhi = hi.copy(); nhi[ax] = ilo[ax]
            out.append((nlo, nhi)); lo[ax] = ilo[ax]
        if hi[ax] > ihi[ax] + 1e-9:               # slab above the crop on this axis
            nlo = lo.copy(); nhi = hi.copy(); nlo[ax] = ihi[ax]
            out.append((nlo, nhi)); hi[ax] = ihi[ax]
    return out                                    # the leftover == intersection, dropped


def clip_boxes_to_crop(boxes, crop_remove, min_dim=0.005):
    """Carve every CROP_REMOVE region out of every box, so a box spanning from
    below (base) to above (bed) the crop no longer covers the empty band between.
    Sub-boxes thinner than min_dim on any axis are dropped."""
    if not crop_remove:
        return boxes
    crops = _crop_bounds(crop_remove)
    out = []
    for b in boxes:
        c = np.array(b['center']); h = np.array(b['size']) / 2.0
        pieces = [(c - h, c + h)]
        for clo, chi in crops:
            nxt = []
            for plo, phi in pieces:
                nxt += _box_minus(plo, phi, clo, chi)
            pieces = nxt
        for plo, phi in pieces:
            if np.all((phi - plo) >= min_dim):
                out.append({'center': [float(x) for x in (plo + phi) / 2.0],
                            'size': [float(x) for x in (phi - plo)],
                            'rpy': [0.0, 0.0, 0.0]})
    return out


def crop_violations(boxes, crop_remove, tol=1e-4):
    """Number of boxes that still overlap any crop region (must be 0 after clip)."""
    if not crop_remove:
        return 0
    crops = _crop_bounds(crop_remove)
    n = 0
    for b in boxes:
        c = np.array(b['center']); h = np.array(b['size']) / 2.0
        blo, bhi = c - h, c + h
        for clo, chi in crops:
            if np.all(np.minimum(bhi, chi) - np.maximum(blo, clo) > tol):
                n += 1
                break
    return n


def coverage_report(pts, verts, boxes):
    """Fraction of an INDEPENDENT validation point sample inside the box union,
    and the max protrusion (mm) of any point outside all boxes. Boxes are
    axis-aligned in local frame (rpy currently 0), so membership is a simple
    interval test. Using a separate validation sample (not the points the boxes
    were built from) makes the coverage number an honest check rather than
    trivially 100%."""
    inside = np.zeros(len(pts), dtype=bool)
    # signed outside-distance per box; min across boxes = distance to the union
    out_dist = np.full(len(pts), np.inf)
    for b in boxes:
        c = np.array(b['center'])
        h = np.array(b['size']) / 2.0
        d = np.abs(pts - c) - h                       # per-axis outside margin
        inside |= np.all(d <= 0, axis=1)
        out_dist = np.minimum(out_dist, np.linalg.norm(np.clip(d, 0, None), axis=1))

    coverage = float(inside.mean())
    max_protrusion_mm = float(out_dist[~inside].max() * 1000.0) if (~inside).any() else 0.0

    # over-approximation indicator: box-union volume vs. mesh AABB volume
    box_vol = float(sum(np.prod(b['size']) for b in boxes))
    aabb = verts.max(axis=0) - verts.min(axis=0)
    aabb_vol = float(np.prod(aabb))
    return {
        'voxel_coverage': round(coverage, 4),
        'max_protrusion_mm': round(max_protrusion_mm, 2),
        'box_union_volume_m3': round(box_vol, 6),
        'mesh_aabb_volume_m3': round(aabb_vol, 6),
        'fill_ratio_vs_aabb': round(box_vol / aabb_vol, 3) if aabb_vol > 0 else None,
        'num_boxes': len(boxes),
    }


def save_preview(verts, boxes, out_png, name):
    """matplotlib 3D: mesh point cloud (light) + box wireframes (red). Headless
    Agg backend, no display / EGL needed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    # downsample points for a legible, fast render
    idx = np.random.default_rng(0).choice(len(verts), size=min(8000, len(verts)),
                                          replace=False)
    p = verts[idx]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, c='0.6', alpha=0.35)

    edges_idx = [(0, 1), (1, 3), (3, 2), (2, 0),
                 (4, 5), (5, 7), (7, 6), (6, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
    for b in boxes:
        c = np.array(b['center'])
        h = np.array(b['size']) / 2.0
        corners = np.array([[sx, sy, sz] for sx in (-h[0], h[0])
                            for sy in (-h[1], h[1]) for sz in (-h[2], h[2])]) + c
        segs = [[corners[i], corners[j]] for i, j in edges_idx]
        ax.add_collection3d(Line3DCollection(segs, colors='red', linewidths=1.2))

    # equal aspect
    allpts = np.vstack([p] + [np.array(b['center']) for b in boxes])
    rng = (allpts.max(axis=0) - allpts.min(axis=0)).max() / 2.0
    mid = (allpts.max(axis=0) + allpts.min(axis=0)) / 2.0
    ax.set_xlim(mid[0] - rng, mid[0] + rng)
    ax.set_ylim(mid[1] - rng, mid[1] + rng)
    ax.set_zlim(mid[2] - rng, mid[2] + rng)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (-Y = door side)'); ax.set_zlabel('Z (m)')
    ax.set_title(f'{name}: mesh vs {len(boxes)} collision boxes')
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    # ================= CONFIG (edit these; no command-line args) =============
    PRINTER = 'a1_mini'   # output basename + KNOWN_DIMS key ('a1' or 'a1_mini')
    MESH = 'models/printers/bambulab_a1_mini.stp'   # STEP/STL/OBJ (rel to repo or abs)
    SCALE = 0.001         # mesh units -> meters (0.001 = mm; 1.0 if already m)
    ROTATE_Z_DEG = 0.0    # yaw the mesh so the operator/door side faces -Y

    # --- decomposition method ---
    # 'coacd' : approximate convex decomposition -> box per piece. Fewer, tighter
    #           boxes (a1_mini: ~18-30 boxes vs voxel's 44). Needs coacd+skimage.
    # 'voxel' : greedy voxel box cover (dep-light fallback).
    METHOD = 'coacd'
    COACD_THRESHOLD = 0.12   # concavity: lower -> more/tighter pieces (0.05..0.2)
    REMESH_MM = 6.0          # marching-cubes remesh resolution before CoACD
    # voxel-method knobs (only used when METHOD='voxel'):
    VOXEL_MM = 20.0
    MIN_FILL = 1.0
    DILATE = 1               # conservative margin (voxels) for both methods
    # ------------------------------------------------------------------------
    # CROP_REMOVE: axis-aligned boxes [cx,cy,cz, sx,sy,sz] (meters, printer-center
    # frame) whose points are deleted BEFORE decomposition. Use to clear a lift
    # corridor / remove the low print head so the pickup can raise the plate up
    # the central bay. Preview the mesh frame with tools/view_printer_model.py.
    CROP_REMOVE = []
    # ------------------------------------------------------------------------
    # A1 preset — the mesh was hand-edited to keep only the desired parts (gantry
    # removed for the pickup model), so NO crop is needed:
    #   PRINTER='a1'
    #   MESH='models/printers/Bambulab A1 printer, modified.stl'
    #   CROP_REMOVE=[]
    # ========================================================================

    mesh_path = MESH if os.path.isabs(MESH) else os.path.join(REPO_ROOT, MESH)
    if not os.path.exists(mesh_path):
        print(DOWNLOAD_HINT.format(path=mesh_path), file=sys.stderr)
        return 2

    mesh, verts = load_mesh(mesh_path, SCALE, ROTATE_Z_DEG)

    dims = verts.max(axis=0) - verts.min(axis=0)
    print(f"[{PRINTER}] loaded {mesh_path}")
    print(f"  mesh AABB (m): W={dims[0]:.3f} D={dims[1]:.3f} H={dims[2]:.3f}")
    if PRINTER in KNOWN_DIMS:
        kf = KNOWN_DIMS[PRINTER]['footprint']
        print(f"  published dims (m): W={kf[0]:.3f} D={kf[1]:.3f} H={kf[2]:.3f}  "
              f"(sanity check — adjust SCALE/ROTATE_Z_DEG if these disagree)")

    pts_build = sample_points(mesh, 300000)
    pts_val = sample_points(mesh, 200000)          # independent validation set

    # crop out points inside any CROP_REMOVE box (e.g. the low print head)
    if CROP_REMOVE:
        def _keep(P):
            drop = np.zeros(len(P), bool)
            for cx, cy, cz, sx, sy, sz in CROP_REMOVE:
                c = np.array([cx, cy, cz]); h = np.array([sx, sy, sz]) / 2.0
                drop |= np.all(np.abs(P - c) <= h, axis=1)
            return P[~drop]
        n0 = len(pts_build)
        pts_build, pts_val = _keep(pts_build), _keep(pts_val)
        print(f"  cropped {len(CROP_REMOVE)} region(s): removed {n0 - len(pts_build)} build pts")

    if METHOD == 'coacd':
        boxes, stats = decompose_coacd(pts_build, remesh_mm=REMESH_MM, dilate=DILATE,
                                       threshold=COACD_THRESHOLD)
    else:
        boxes, stats = decompose_boxes(pts_build, voxel_mm=VOXEL_MM, dilate=DILATE,
                                       min_fill=MIN_FILL)

    # carve the crop regions out of the FINAL boxes: point-cropping alone leaves
    # boxes that span from below to above the gap, still covering it
    if CROP_REMOVE:
        boxes = clip_boxes_to_crop(boxes, CROP_REMOVE)
        viol = crop_violations(boxes, CROP_REMOVE)
        stats['num_boxes'] = len(boxes)
        stats['crop_violations'] = viol
        print(f"  crop check: {viol} box(es) overlap the removed region "
              f"(must be 0){'  <-- FAIL' if viol else '  OK'}")

    cov = coverage_report(pts_val, verts, boxes)
    cov.update(stats)

    print(f"  -> {cov['num_boxes']} boxes ({stats.get('method','voxel')}) "
          f"| coverage={cov['voxel_coverage']:.3f} "
          f"| max_protrusion={cov['max_protrusion_mm']}mm")

    bed = KNOWN_DIMS.get(PRINTER, {}).get('bed')
    model = {
        'name': PRINTER,
        'source_mesh': os.path.relpath(mesh_path, REPO_ROOT),
        'units_scale': SCALE,
        'rotate_z_deg': ROTATE_Z_DEG,
        'frame': 'printer_center',
        'footprint': {'width': float(dims[0]), 'depth': float(dims[1]),
                      'height': float(dims[2])},
        'bed_size': list(bed) if bed else None,
        'boxes': boxes,
        'coverage': cov,
    }

    os.makedirs(PRINTERS_DIR, exist_ok=True)
    out_json = os.path.join(PRINTERS_DIR, f'{PRINTER}.json')
    with open(out_json, 'w') as f:
        json.dump(model, f, indent=2)
    print(f"  wrote {os.path.relpath(out_json, REPO_ROOT)}")

    out_png = os.path.join(PRINTERS_DIR, f'{PRINTER}_preview.png')
    try:
        save_preview(pts_build, boxes, out_png, PRINTER)
        print(f"  wrote {os.path.relpath(out_png, REPO_ROOT)}")
    except Exception as e:
        print(f"  (preview render skipped: {e})", file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
