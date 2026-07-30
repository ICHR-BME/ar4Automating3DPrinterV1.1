#!/usr/bin/env python3
"""
Scan for 3 markers (IDs 0, 1, 2) then start the interactive command menu.

Set runVirtual = 1 in main() to run against Gazebo (start it first with
scripts/launchVirtualRobot.sh) instead of the physical robot + webcam.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy

from ar4_automation.marker_sources import MANUAL
from ar4_automation.runner_common import (
    make_sim_node,
    make_sim_printer,
    make_webcam_node,
    spawn_printers_from_markers,
    spawn_sim_printers,
    spin_in_background,
    wait_for_joint_states,
    run_command_menu,
    sim_printer_specs,
    register_manual_estimates,
)


def main():
    rclpy.init()
    runVirtual = 0    # 1 = run in Gazebo (sim camera + spawned printers), 0 = hardware
    robot = 'xarm6'   # 'ar4' | 'lite6' | 'xarm6' (sim launch: launchVirtualRobot.sh / launchVirtualXArmLite6.sh / launchVirtualXArm6.sh)
    # 1 = hardware initial estimates come from data/manual_marker_estimates.json
    # (written by teachMarkersByHand.py: drag-teach the arm until the camera
    # sees each marker); markers not in that file keep the geometric estimates
    # from the printer models below
    use_manual_estimates = 1

    if runVirtual:
        # make_sim_node, not a bare printerAutomation: it carries the shared
        # marker_sizes (a sim node built without them fell back to the detector's
        # old 0.05 default and reported every marker twice as far as it was) and
        # the per-robot gripper_disabled rule.
        node = make_sim_node(robot=robot)
        node.randomize_estimated_markers = True
    else:
        node = make_webcam_node(robot=robot)

    # spin before spawning printers: their TF lookups need the background
    # executor (spin_once starves on the 30 Hz camera callbacks otherwise)
    spin_in_background(node)
    wait_for_joint_states(node)

    # Printer layout: BODY poses (the box model's center) in the good frame, one
    # entry per printer, with the ArUco ID its 'door' mount wears. Sim uses the
    # per-robot layout from runner_common; hardware uses the measured bench
    # layout below. printer_model names a box model in models/printers/.
    hardware_specs = [
        {"marker_id": 0, "pos": [0.28, -0.3, 0.065], "orient": [0.0, 0.0, np.pi]},
        {"marker_id": 1, "pos": [0.50, -0.3, 0.065], "orient": [0.0, 0.0, np.pi]},
        {"marker_id": 2, "pos": [0.65, 0.1, 0.075], "orient": [0.0, 0.0, 3/2*np.pi]},
    ]
    specs = sim_printer_specs(robot, 3) if runVirtual else hardware_specs

    if runVirtual:
        if use_manual_estimates:
            # spawn the printers where the hand-taught markers say they are, so
            # what the scan looks for and what stands in Gazebo agree
            printer1, printer2, printer3 = spawn_printers_from_markers(
                node, specs, source=MANUAL)
        else:
            printer1, printer2, printer3 = spawn_sim_printers(node, specs)
    else:
        # hardware: nothing is spawned; these only supply geometric seeds for
        # the markers the manual file doesn't cover
        printer1, printer2, printer3 = [make_sim_printer(node, s) for s in specs]

    node.get_logger().info("Starting initial scan for markers...")
    node.load_state()
    # markers are pinned by default — each only updates during its own scan
    # windows, so menu scrapes can't drift the scrape marker between runs

    node.marker_offset_config[0] = 'box_offset'
    node.marker_offset_config[1] = 'box_offset'
    node.marker_offset_config[2] = 'printer_offset'

    # register the initial door-marker estimates (after load_state so stale
    # saved poses can't shadow them), then scan all markers: hand-taught
    # estimates where available, geometric estimates from the printer models
    # for the rest
    manual_ids = []
    if not runVirtual and use_manual_estimates:
        manual_ids = register_manual_estimates(node)

    for printer in (printer1, printer2, printer3):
        # geometric seed: where this printer's 'door' mount sits, given the body
        # pose above. Skipped for markers the hand-taught file already covered.
        if set(printer.marker_ids.values()) & set(manual_ids):
            continue
        printer.register_marker_estimates(node)

    # save the layout alongside the markers so restore_saved_printers can rebuild
    # these printers in a later session
    node.register_printers(specs)

    viewing_distance = 0.15
    node.scanMarkerApproach(marker_id=0, viewing_distance=viewing_distance)
    node.scanMarkerApproach(marker_id=1, viewing_distance=viewing_distance)
    node.scanMarkerApproach(marker_id=2, viewing_distance=viewing_distance)

    node.get_logger().info("Initial scan complete.")
    run_command_menu(node)


if __name__ == '__main__':
    main()
