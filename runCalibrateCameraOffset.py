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

from ar4_automation.runner_common import (
    start_node,
    restore_saved_printers,
    spawn_sim_printers,
    sim_printer_specs,
)

# ---- Configuration ----
RUN_SIM          = 0          # 1 = Gazebo (sim camera + spawned printers), 0 = hardware
ROBOT            = 'lite6'    # 'ar4' | 'lite6' | 'xarm6'
MARKER_ID         = 1                    # the stationary marker to calibrate against
VIEWING_DISTANCES = (0.25, 0.15)   # camera standoffs (m), swept FAR->NEAR
APPLY             = 1                     # 1 = install the correction this session, 0 = report only


def main():
    rclpy.init()
    node = start_node(sim=RUN_SIM, robot=ROBOT)
    speed_scale = 0.2
    node.moveit2.max_velocity = speed_scale
    node.moveit2.max_acceleration = speed_scale
    if RUN_SIM:
        spawn_sim_printers(node, sim_printer_specs(ROBOT, 2))
    else:
        if not node.load_state():
            node.get_logger().error("No save file found — run printer_automation.py first to create one.")
            return
        restore_saved_printers(node)
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
