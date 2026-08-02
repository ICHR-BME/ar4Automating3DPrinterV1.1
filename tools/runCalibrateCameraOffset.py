#!/usr/bin/env python3
"""
Estimate the eef->camera mount translation error (translation-only, eye-in-hand)
from a single stationary marker, by orbiting the camera around it and checking
the mismatch between the (constant) true marker position and the per-pose
measured positions.

Reports the correction delta and the corrected mount translation. Set APPLY = 1
to install it for the session (fixes both scan targeting and measured marker
poses so a subsequent pickup uses it); the URDF is never modified — for a
permanent fix, shift the camera mount origin in the URDF by delta (EEF frame).

Set RUN_SIM = 1 for Gazebo (start scripts/launchVirtualRobot.sh first).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy

from ar4_automation.marker_sources import SCAN
from ar4_automation.runner_common import (
    start_node,
    restore_saved_printers,
    spawn_printers_from_markers,
    spawn_sim_printers,
    sim_printer_specs,
    register_manual_estimates,
)

# ---- Configuration ----
RUN_SIM          = 0          # 1 = Gazebo (sim camera + spawned printers), 0 = hardware
ROBOT            = 'xarm6'    # 'ar4' | 'lite6' | 'xarm6'
MARKER_ID         = 1                    # the stationary marker to calibrate against
VIEWING_DISTANCES = (0.25, 0.15)   # camera standoffs (m), swept FAR->NEAR
APPLY             = 1                     # 1 = install the correction this session, 0 = report only
# 1 = seed the initial scan of MARKER_ID from data/manual_marker_estimates.json
# (written by teachMarkersByHand.py), overriding any saved pose; with it a
# save file is optional
USE_MANUAL_ESTIMATES = 1
# 1 = sim printers stand where the last scan MEASURED their markers
# (data/printer_state.json, written by scanFor2Markers.py), so Gazebo shows the same
# layout the arm is working from. 0 = spawn them at the runner_common layout and
# take the marker estimates from there instead — no scan needed, but the save
# file is then ignored and the scene won't match a scanned run.
SPAWN_FROM_SCAN = 1
# 1 = collisions on, in BOTH places they exist: the MoveIt planning scene (a
# ground plane under the base plus a box model of every printer) and Gazebo
# physics (printers spawn with real <collision> geometry). 0 = turn off both —
# the planning scene is emptied, including anything an earlier run left in
# move_group, and printers spawn visual-only so the arm passes through them
# instead of stalling against them. Self-collisions and joint limits are ALWAYS
# enforced. Use 0 to tell "the plan is in collision" apart from "the goal is
# unreachable", not for a real run.
COLLISIONS      = 1


def main():
    rclpy.init()
    node = start_node(sim=RUN_SIM, robot=ROBOT, collisions=COLLISIONS)
    speed_scale = 0.2
    node.moveit2.max_velocity = speed_scale
    node.moveit2.max_acceleration = speed_scale
    if RUN_SIM and not SPAWN_FROM_SCAN:
        # the spawned printers ARE the ground truth here: marker estimates are
        # derived from where they were placed, and the save file is ignored
        spawn_sim_printers(node, sim_printer_specs(ROBOT, 2))
    else:
        have_save = node.load_state()
        if have_save and RUN_SIM:
            # stand each printer where the scan measured its marker (nothing is
            # re-registered, so the scanned poses stay as saved)
            spawn_printers_from_markers(node, sim_printer_specs(ROBOT, 2),
                                        source=SCAN, fallback=False)
        elif have_save:
            restore_saved_printers(node)
        # hand-taught estimates (after load_state so stale saved poses can't
        # shadow them) — the official scanToMarker below re-measures anyway
        manual_ids = []
        if USE_MANUAL_ESTIMATES:
            manual_ids = register_manual_estimates(node, marker_ids=[MARKER_ID])
        if not have_save and MARKER_ID not in manual_ids:
            node.get_logger().error(
                f"No save file and no hand-taught estimate for marker {MARKER_ID} — "
                f"run scanFor2Markers.py or teachMarkersByHand.py first."
            )
            return
        # NB: do NOT lock MARKER_ID — calibration must re-measure it across poses.

    # center on the marker first, from the farthest standoff so it's safely
    # in-FOV before the closer orbit poses run
    node.scanToMarker(marker_id=MARKER_ID, viewing_distance=max(VIEWING_DISTANCES))

    delta = node.calibrate_camera_offset(
        marker_id=MARKER_ID,
        viewing_distances=VIEWING_DISTANCES,
        apply=bool(APPLY),
    )
    if delta is None:
        node.get_logger().error("calibrate_camera_offset failed — see log above.")
    else:
        node.get_logger().info(
            f"calibrate_camera_offset done. delta(EEF,m)={[round(float(d),4) for d in delta]}, "
            f"applied={bool(APPLY)}."
        )


if __name__ == '__main__':
    main()
