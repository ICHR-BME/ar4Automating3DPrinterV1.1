#!/usr/bin/env python3
"""
Scan for markers 1 and 2 only, then start the interactive command menu.

Set runVirtual = 1 in main() to run against Gazebo 
(start it first with
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
    runVirtual = 1    # 1 = run in Gazebo (sim camera + spawned printers), 0 = hardware
    robot = 'xarm6'     # 'ar4' | 'lite6' | 'xarm6' (sim launch: launchVirtualRobot.sh / launchVirtualXArmLite6.sh / launchVirtualXArm6.sh)
    # 1 = initial estimates come from data/manual_marker_estimates.json
    # (written by teachMarkersByHand.py: drag-teach the arm until the camera
    # sees each marker) instead of the geometric seeds below. Applies in sim as
    # well as on hardware — in sim that means the scan starts from the measured
    # poses rather than from where the printers were spawned, which is how you
    # rehearse the hardware flow without the arm.
    use_manual_estimates = 1
    if runVirtual:
        # make_sim_node, not a bare printerAutomation: it carries the shared
        # marker_sizes (a sim node built without them fell back to the detector's
        # old 0.05 default and reported every marker twice as far as it was) and
        # the per-robot gripper_disabled rule.
        node = make_sim_node(robot=robot)
        node.randomize_estimated_markers = False
    else:
        node = make_webcam_node(robot=robot)

    # Temporarily slow the arm for this session. These are MoveIt scaling
    # factors (0-1) on the robot's configured joint vel/accel limits; the
    # stack default is 0.9 (pose_reader.py). Applies to every planned move
    # here (go_home still overrides with its own velocity_scaling arg, then
    # restores to this value).
    speed_scale = 0.7
    node.moveit2.max_velocity = speed_scale
    node.moveit2.max_acceleration = speed_scale

    # spin before spawning printers: their TF lookups need the background
    # executor (spin_once starves on the 30 Hz camera callbacks otherwise)
    spin_in_background(node)
    wait_for_joint_states(node)

    # Printer layout: BODY poses (the box model's center) in the good frame, one
    # entry per printer, with the ArUco ID its 'door' mount wears. Sim uses the
    # per-robot layout from runner_common; hardware uses the measured bench
    # layout below. printer_model names a box model in models/printers/.
    hardware_specs = [
        {"marker_id": 1, "pos": [0.40, 0.10, -0.1],
         "orient": [-1/2*np.pi, 0.0, 2/2*np.pi], "printer_model": 'a1'},
        {"marker_id": 2, "pos": [0.30, 0.10, 0.1],
         "orient": [0.0, 0.0, 1*np.pi], "printer_model": 'a1_mini'},
    ]
    specs = sim_printer_specs(robot, 2) if runVirtual else hardware_specs

    if runVirtual:
        if use_manual_estimates:
            # spawn the printers where the hand-taught markers say they are, so
            # what the scan looks for and what stands in Gazebo agree
            printer2, printer3 = spawn_printers_from_markers(
                node, specs, source=MANUAL)
        else:
            printer2, printer3 = spawn_sim_printers(node, specs)
    else:
        # hardware: nothing is spawned; these only supply geometric seeds for
        # the scan when no marker file is used (see use_manual_estimates)
        printer2, printer3 = [make_sim_printer(node, s) for s in specs]

    node.get_logger().info("Starting initial scan for markers 1 and 2...")
    node.load_state()
    # markers are pinned by default — each only updates during its own scan
    # windows, so menu scrapes can't drift the scrape marker between runs

    node.marker_offset_config[1] = 'box_offset'
    node.marker_offset_config[2] = 'printer_offset'

    # register the initial door-marker estimates (after load_state so stale
    # saved poses can't shadow them), then scan both markers
    manual_ids = []
    if use_manual_estimates:
        # hand-taught estimates from teachMarkersByHand.py. Not gated on
        # runVirtual: the file is just measured marker poses in base_link, which
        # seed a sim scan the same way they seed a hardware one. A missing or
        # unreadable file returns [] and falls through to the seeds below.
        manual_ids = register_manual_estimates(node)

    if not manual_ids:
        # geometric estimates: where each printer's 'door' mount sits, given the
        # body poses above
        printer2.register_marker_estimates(node)
        printer3.register_marker_estimates(node)

    # save the body poses alongside the markers, so restore_saved_printers can
    # rebuild these printers in a later session
    node.register_printers(specs)


    viewing_distance = 0.15
    node.scanMarkerApproach(marker_id=1, viewing_distance=viewing_distance)
    node.scanMarkerApproach(marker_id=2, viewing_distance=viewing_distance)

    node.get_logger().info("Initial scan complete.")

    # persist markers, offset config, and printer configs immediately (same as
    # runDoubleTransfer.py) so the run* scripts can load them instead of
    # re-scanning. Don't rely on the 5s auto-save timer alone — if the session
    # ends before it fires, printer_state.json is left empty.
    node.save_state()
    node.get_logger().info("Saved marker/printer state to printer_state.json")

    run_command_menu(node)


if __name__ == '__main__':
    main()
