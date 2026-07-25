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

from ar4_automation.printer_automation import printerAutomation
from ar4_automation.runner_common import (
    make_webcam_node,
    spin_in_background,
    wait_for_joint_states,
    run_command_menu,
    sim_printer_specs,
    register_manual_estimates,
)
from ar4_automation.simulated3DPrinter import Simulated3DPrinter


def main():
    rclpy.init()
    runVirtual = 0    # 1 = run in Gazebo (sim camera + spawned printers), 0 = hardware
    robot = 'xarm6'     # 'ar4' | 'lite6' | 'xarm6' (sim launch: launchVirtualRobot.sh / launchVirtualXArmLite6.sh / launchVirtualXArm6.sh)
    # 1 = hardware initial estimates come from data/manual_marker_estimates.json
    # (written by teachMarkersByHand.py: drag-teach the arm until the camera
    # sees each marker) instead of the hardcoded printer coordinates below
    use_manual_estimates = 1
    if runVirtual:
        node = printerAutomation(calibration_mode=False, stream_source="ros", robot=robot)
        node.gripper_disabled = True
        node.randomize_estimated_markers = False
    else:
        node = make_webcam_node(robot=robot)

    # Temporarily slow the arm for this session. These are MoveIt scaling
    # factors (0-1) on the robot's configured joint vel/accel limits; the
    # stack default is 0.9 (pose_reader.py). Applies to every planned move
    # here (go_home still overrides with its own velocity_scaling arg, then
    # restores to this value).
    speed_scale = 0.3
    node.moveit2.max_velocity = speed_scale
    node.moveit2.max_acceleration = speed_scale

    # spin before spawning printers: their TF lookups need the background
    # executor (spin_once starves on the 30 Hz camera callbacks otherwise)
    spin_in_background(node)
    wait_for_joint_states(node)

    if runVirtual:
        # per-robot layout from runner_common (positions in the good frame)
        printer2, printer3 = [
            Simulated3DPrinter(node=node, pos=s["pos"], orient=s["orient"],
                               door_marker_texture=s["door_marker_texture"])
            for s in sim_printer_specs(robot, 2)
        ]
        printer2.spawn_fast()
        printer3.spawn_fast()
    else:
        '''
        printer2 = Simulated3DPrinter(
            node=node, pos=[0.40, -0.3, 0.065], orient=[0.0, 0.0, np.pi],
            door_marker_texture='materials/textures/marker6x6_1.png',
        )
        printer3 = Simulated3DPrinter(
            node=node, pos=[0.65, 0.1, 0.075], orient=[0.0, 0.0, 3/2*np.pi],
            door_marker_texture='materials/textures/marker6x6_2.png',
        )
        '''
        
        printer2 = Simulated3DPrinter(
            node=node, pos=[0.40, 0.10, -0.1], orient=[-1/2*np.pi, 0.0, 2/2*np.pi],
            door_marker_texture='materials/textures/marker6x6_1.png',
        )
        printer3 = Simulated3DPrinter(
            node=node, pos=[0.30, 0.1, 0.1], orient=[0.0, 0.0, 1*np.pi],
            door_marker_texture='materials/textures/marker6x6_2.png',
        )
        '''printer3 = Simulated3DPrinter(
            node=node, pos=[0.35, -0.3, 0.15], orient=[0.0, 0.0, -1/2*np.pi],
            door_marker_texture='materials/textures/marker6x6_2.png',
        )'''
        '''printer2 = Simulated3DPrinter(
                    node=node, pos=[0.1, -0.6, 0.25], orient=[0.0, 0.0, 1*np.pi],
                    door_marker_texture='materials/textures/marker6x6_1.png',
                )'''

    node.get_logger().info("Starting initial scan for markers 1 and 2...")
    node.load_state()
    # markers are pinned by default — each only updates during its own scan
    # windows, so menu scrapes can't drift the scrape marker between runs

    node.marker_offset_config[1] = 'box_offset'
    node.marker_offset_config[2] = 'printer_offset'

    # register the initial door-marker estimates (after load_state so stale
    # saved poses can't shadow them), then scan both markers
    manual_ids = []
    if not runVirtual and use_manual_estimates:
        # hand-taught estimates from teachMarkersByHand.py
        manual_ids = register_manual_estimates(node)

    if not manual_ids:
        # geometric estimates from the printer models above
        bad_pos, bad_euler = printer2.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=1, bad_pos=bad_pos, bad_euler=bad_euler)
        bad_pos, bad_euler = printer3.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=2, bad_pos=bad_pos, bad_euler=bad_euler)

    '''if not runVirtual:
        node.register_printers([
            {"marker_id": 2, "pos": [0.3, -0.45, 0.24], "orient": [0.0, 0.0, 0*np.pi],
                "door_marker_texture": 'materials/textures/marker6x6_2.png'},
            {"marker_id": 1, "pos": [0.75, 0.1, 0.07], "orient": [0.0, 0.0, 3/2*np.pi],
                "door_marker_texture": 'materials/textures/marker6x6_1.png'},
        ])
        '''''''node.register_printers([
            {"marker_id": 1, "pos": [0.56, -0.35, 0.07], "orient": [0.0, 0.0, np.pi],
             "door_marker_texture": 'materials/textures/marker6x6_1.png'},
            {"marker_id": 2, "pos": [0.75, 0.1, 0.07], "orient": [0.0, 0.0, 3/2*np.pi],
             "door_marker_texture": 'materials/textures/marker6x6_2.png'},
        ])'''
    
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
