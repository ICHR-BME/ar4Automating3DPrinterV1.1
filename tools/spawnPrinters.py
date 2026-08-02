#!/usr/bin/env python3
"""Spawn printer models into a running Gazebo, placed from a marker file.

Launch a sim first (any gz world with a /world/<WORLD>/create service — e.g.
scripts/launchVirtualXArm6.sh), then run this. Each printer is one SDF model:
the collision boxes from models/printers/<name>.json plus a textured plate for
every ArUco mount in that JSON.

Config-driven — edit the CONFIG block below (no command-line args).

Where the printers go (MARKER_SOURCE):
  'scan'   data/printer_state.json, written by scanFor2Markers.py /
           scanFor3Markers.py — the poses a real camera scan measured.
  'manual' data/manual_marker_estimates.json, written by teachMarkersByHand.py
           — drag-teach measurements, useful before a scan has ever run.
  'none'   ignore both files and use the fallback poses in PRINTERS.
For each printer, the body is back-solved from its marker plus the mount offset
recorded in the model JSON (PrinterModel.pose_from_marker). Markers missing from
the file fall back to the PRINTERS pose below.

Notes:
- The box models are already Z-up (printer_mesh_to_boxes recenters the AABB and
  does no axis swap — a1's Z extent is its 0.605 m height), so the fallback
  orientation is identity, not a corrective roll. Each fallback z = half the
  model's HEIGHT so the base rests at z=0. A pose derived from a marker carries
  its own orientation.
"""

import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from ar4_automation.marker_sources import load_marker_poses, source_path
from ar4_automation.printer_model import PrinterModel
from ar4_automation.simulated3DPrinter import Simulated3DPrinter


def main():
    # ===================== CONFIG (edit these) =====================
    WORLD = 'default'          # gz world name (matches /world/<WORLD>/create)
    UPRIGHT = [0.0, 0.0, 0.0]  # box models are already Z-up; no corrective roll

    # which file the printer poses come from: 'scan', 'manual', or 'none'
    # (or a path to a file in either format)
    MARKER_SOURCE = 'scan'
    # 'scan' only: True = ignore markers flagged 'estimated', i.e. seeds a scan
    # never actually confirmed. The 'manual' file is all measurements, so this
    # has no effect there.
    REQUIRE_DETECTED = True
    # which mount the marker pose refers back to (a key in the model JSON's
    # "markers", placed with tools/view_printer_model.py)
    ANCHOR_MARKER = 'door'

    # (printer_model, fallback [x, y, z] m, {marker_name: aruco_id}).
    # The IDs are what the printer WEARS: they pick the spawned texture, and
    # they're how a pose from the marker file is matched back to a printer. Two
    # printers must NOT share an ID or the detector can't tell them apart.
    # Fallback z = half the model's height -> base on floor:
    #   a1 h=0.605 -> z=0.302 ; a1_mini h=0.385 -> z=0.193
    PRINTERS = [
        ('a1',      [0.60,  0.30, 0.302], {'door': 1}),
        ('a1_mini', [0.60, -0.30, 0.193], {'door': 2}),
    ]
    # 1 = spawn solid printers (real <collision> geometry, the arm stalls against
    # them). 0 = visual-only boxes the arm passes straight through — the same
    # switch the runner scripts' COLLISIONS var flips, for a scene spawned by
    # hand. They are <static> either way, so nothing falls.
    COLLIDE = 1
    # ===============================================================

    rclpy.init()
    node = Node('spawn_printers')
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    time.sleep(0.3)

    poses = load_marker_poses(MARKER_SOURCE, require_detected=REQUIRE_DETECTED,
                              log=node.get_logger().info)
    src = os.path.basename(source_path(MARKER_SOURCE))

    printers = []
    for name, fallback_pos, marker_ids in PRINTERS:
        model = PrinterModel.load(name)
        anchor_id = marker_ids.get(ANCHOR_MARKER)
        t0 = time.time()

        if anchor_id in poses:
            # back-solve the body pose from the marker: the mount offset in the
            # model JSON is what turns a marker observation into a printer pose
            p = Simulated3DPrinter.from_marker(
                *poses[anchor_id], printer_model=model, mount=ANCHOR_MARKER,
                node=node, marker_ids=marker_ids, world_name=WORLD,
                collide=COLLIDE)
            source = f'marker {anchor_id} from {src}'
        else:
            p = Simulated3DPrinter(
                node=node, pos=fallback_pos, orient=UPRIGHT, printer_model=model,
                marker_ids=marker_ids, world_name=WORLD, use_bad_frame=False,
                collide=COLLIDE)
            source = 'fallback pose'

        ok = p.spawn()
        # sanity flag: a derived pose is only as good as the mount offset, and a
        # mount clicked at the wrong height shifts the whole body by that error.
        # Report where the base lands so a printer floating above (or sunk into)
        # the floor is obvious.
        base_z = float(p.pos[2]) - model.height / 2.0
        node.get_logger().info(
            f"'{p.name}' [{source}] in {time.time()-t0:.2f}s "
            f"{'OK' if ok else 'FAILED'}; base sits at z={base_z:+.3f} m "
            f"(0 = on the floor). Its {ANCHOR_MARKER} marker should be seen at "
            f"{[round(float(v), 4) for v in p.marker_pose_in_base(ANCHOR_MARKER)[0]]}")
        printers.append(p)

    node.get_logger().info(
        "Done. Look in Gazebo for the printers. Ctrl-C to exit "
        "(they stay spawned).")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
