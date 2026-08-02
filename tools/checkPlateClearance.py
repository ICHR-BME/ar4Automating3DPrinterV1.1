#!/usr/bin/env python3
"""Pre-flight check: will the held build plate clear a printer's collision
boxes at every waypoint of its procedures?

Walks each procedure list in models/printers/<PRINTER>.json exactly the way
printerAutomation will — tracking WHEN the plate is attached ('held_plate'
entries) and WHICH boxes are solid ('entry_zone_collisions' entries) — and
sweeps the plate's volume (inflated by MARGIN) along the motion between
consecutive waypoints. Any solid box the plate touches is reported with its
index, so a run that would die with an in-collision goal (planner FAILURE with
no useful message) is caught here, with names, in a second.

This is the offline twin of the live failure mode: after re-running
printer_mesh_to_boxes.py the box indices change and entry-zone flags are
dropped, so a set-down pose that used to clear can silently gain a solid box
(seen live: one un-flagged rail beside the A1's bed made every place descend
unplannable). Run this after every re-boxing / procedure edit.

Everything is computed in the PRINTER-LOCAL frame from the model JSON alone —
the marker mount pins the waypoint frame, the canonical grasp orientation
(same [pi, 0, pi] the runtime uses) pins the tool, and the plate spec pins the
plate. No ROS, no scan file, no sim needed.

Fix suggestions it prints: SHIFT+click the named box in
definePrinterApproximateModel.py (entry zone), raise the waypoint, or shrink
the plate spec.

Configure below (no command-line args) and run:
    python3 tools/checkPlateClearance.py
Exit code 1 if any violation was found.
"""

import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRINTERS_DIR = os.path.join(REPO_ROOT, 'models', 'printers')

# ================= CONFIG (edit these; no command-line args) =================
PRINTER = 'a1'        # whose boxes + procedures to check
# whose 'plate' section the gripper is holding. None = PRINTER's own plate.
# For the scrape fixture set PRINTER='scrape_fixture', PLATE_FROM='a1' (the
# plate comes from the source printer, the boxes from the fixture).
PLATE_FROM = None
PROCEDURES = None     # e.g. ['place'] to limit; None = every list in the JSON
MOUNT = 'door'        # marker mount the waypoints are relative to
MARGIN = 0.01         # inflate the plate this much (m) per side — near-misses
SWEEP_STEPS = 8       # interpolation steps between consecutive waypoints
# =============================================================================

# the runtime's canonical grasp orientation in the marker frame
# (printer_automation.GRASP_ORI_IN_MARKER — kept literal here so this tool
# needs no ROS imports)
GRASP_ORI_IN_MARKER = [np.pi, 0.0, np.pi]
# plate-mount convention -> aruco convention (printer_model.marker_aruco_rotation)
PLATE_TO_ARUCO = np.array([[0.0, 0.0, -1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])


def load_model(name):
    with open(os.path.join(PRINTERS_DIR, f'{name}.json')) as f:
        return json.load(f)


def plate_spec(model):
    """size + grasp offset, mirroring held_plate.plate_spec's fallbacks."""
    plate = model.get('plate') or {}
    if 'size' in plate:
        size = [float(v) for v in plate['size']]
    elif model.get('bed_size'):
        size = [float(model['bed_size'][0]), float(model['bed_size'][1]), 0.003]
    else:
        raise SystemExit(f"model '{model['name']}' has no plate/bed_size")
    off_p = [float(v) for v in plate.get('offset_pos', [0.0, 0.0, size[1] / 2.0])]
    off_r = [float(v) for v in plate.get('offset_rpy', [np.pi / 2.0, 0.0, 0.0])]
    return np.array(size), np.array(off_p), R.from_euler('XYZ', off_r)


def marker_frame(model, mount):
    """(pos, Rotation) of the mount's ARUCO frame in the printer-local frame."""
    for mk in model.get('markers', []):
        if mk.get('name') == mount:
            rot = R.from_euler('xyz', mk['rpy']) * R.from_matrix(PLATE_TO_ARUCO)
            return np.asarray(mk['pos'], dtype=float), rot
    raise SystemExit(f"model '{model['name']}' has no '{mount}' mount")


def plate_points(size, n=13):
    """Sample grid over the (inflated) plate volume, plate-local frame."""
    s = size + 2.0 * MARGIN
    g = np.linspace(-0.5, 0.5, n)
    return np.array([[x * s[0], y * s[1], z * s[2]]
                     for x in g for y in g for z in np.linspace(-0.5, 0.5, 3)])


def plate_pose(mpos, mrot, offset, angle_deg, poff_p, poff_R):
    """Plate (center, Rotation) in printer-local frame for a marker-frame
    waypoint offset + tilt — the same chain the runtime composes."""
    R_mt = R.from_euler('XYZ', GRASP_ORI_IN_MARKER)
    if angle_deg:
        R_mt = R.from_euler('x', np.radians(angle_deg)) * R_mt
    R_pt = mrot * R_mt                       # printer-local -> TCP
    tcp = mpos + mrot.apply(offset)
    return tcp + R_pt.apply(poff_p), R_pt * poff_R


def boxes_hit(center, rot, pts_local, boxes, solid_idx):
    """Indices of solid boxes containing any sampled plate point."""
    pts = center + rot.apply(pts_local)
    hit = []
    for i in solid_idx:
        b = boxes[i]
        c = np.asarray(b['center'], dtype=float)
        h = np.asarray(b['size'], dtype=float) / 2.0
        local = R.from_euler('xyz', b.get('rpy', [0, 0, 0])).inv().apply(pts - c)
        if np.any(np.all(np.abs(local) <= h, axis=1)):
            hit.append(i)
    return hit


def main():
    model = load_model(PRINTER)
    plate_model = load_model(PLATE_FROM) if PLATE_FROM else model
    size, poff_p, poff_R = plate_spec(plate_model)
    mpos, mrot = marker_frame(model, MOUNT)
    boxes = model['boxes']
    zone = {i for i, b in enumerate(boxes) if b.get('entry_zone')}
    pts_local = plate_points(size)

    procs = model.get('procedures') or {}
    # waypoint lists only — skip 'notes' (a list of strings) and null lists
    names = PROCEDURES or [k for k, v in procs.items()
                           if isinstance(v, list)
                           and any(isinstance(e, dict) for e in v)]
    print(f"[{PRINTER}] plate {np.round(size, 3).tolist()} m "
          f"(from '{plate_model['name']}', +{MARGIN * 1000:.0f} mm margin), "
          f"{len(boxes)} boxes, entry zone {sorted(zone) if zone else 'EMPTY'}")

    violations = []
    for name in names:
        entries = procs.get(name)
        if not entries:
            continue
        attached = (name != 'pickup')   # place/scrape start carrying the plate
        zone_open = False
        last = None                     # (offset, angle) of the last pos entry

        def solid():
            return [i for i in range(len(boxes))
                    if not (zone_open and i in zone)]

        def check(offset, angle, where):
            c, rot = plate_pose(mpos, mrot, np.asarray(offset, float),
                                angle, poff_p, poff_R)
            for i in boxes_hit(c, rot, pts_local, boxes, solid()):
                violations.append((name, where, i))

        print(f"\n== {name} ({len(entries)} entries, plate "
              f"{'attached' if attached else 'not attached yet'}) ==")
        for i, wp in enumerate(entries):
            if not isinstance(wp, dict):
                continue
            where = f"entry {i + 1}/{len(entries)}"
            if 'held_plate' in wp:
                attached = str(wp['held_plate']).lower() == 'attach'
                if attached and last:
                    check(*last, f"{where} (plate appears at the last pose)")
            elif 'entry_zone_collisions' in wp:
                zone_open = str(wp['entry_zone_collisions']).lower() == 'off'
                if not zone_open and attached and last:
                    check(*last, f"{where} (zone closes around the plate)")
            elif 'pos' in wp:
                offset = [float(v) for v in wp['pos']]
                angle = float(wp.get('angle_deg') or 0.0)
                if attached:
                    if last:
                        for t in np.linspace(0.0, 1.0, SWEEP_STEPS + 1):
                            o = (1 - t) * np.asarray(last[0]) + t * np.asarray(offset)
                            a = (1 - t) * last[1] + t * angle
                            check(o, a, f"{where} (sweep)")
                    else:
                        check(offset, angle, where)
                last = (offset, angle)

    print()
    if not violations:
        print("CLEAR: the plate touches no solid box anywhere in the "
              "checked procedures.")
        return 0
    seen = {}
    for name, where, bi in violations:
        seen.setdefault((name, bi), []).append(where)
    print(f"{len(seen)} VIOLATION(S):")
    for (name, bi), wheres in sorted(seen.items()):
        b = boxes[bi]
        flag = 'entry_zone' if bi in zone else 'NOT in entry zone'
        print(f"  [{name}] box {bi} ({flag}, size "
              f"{np.round(b['size'], 3).tolist()}) hit at {wheres[0]}"
              + (f" (+{len(wheres) - 1} more)" if len(wheres) > 1 else ""))
    print("\nfixes: SHIFT+click the box into the entry zone in "
          "definePrinterApproximateModel.py; or raise/shift the waypoint; or "
          "shrink the plate spec / MARGIN.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
