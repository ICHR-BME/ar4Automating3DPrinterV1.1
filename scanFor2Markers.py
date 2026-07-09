#!/usr/bin/env python3
"""
Scan for markers 1 and 2 only, then start the interactive command menu.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy

from printer_automation import printerAutomation
from runner_common import (
    make_webcam_node,
    spin_in_background,
    wait_for_joint_states,
    run_command_menu,
)
from simulated3DPrinter import Simulated3DPrinter


def main():
    rclpy.init()
    runVirtual = 0

    if runVirtual:
        node = printerAutomation(calibration_mode=False, stream_source="ros")
        node.gripper_disabled = True
        node.randomize_estimated_markers = True

        printer2 = Simulated3DPrinter(
            node=node, pos=[0.44, -0.2, 0.21], orient=[0.0, 0.1, np.pi],
            door_marker_texture='materials/textures/marker6x6_1.png',
        )
        printer2.spawn_fast()

        printer3 = Simulated3DPrinter(
            node=node, pos=[0.60, 0.1, 0.21], orient=[0.0, 0.0, 3/2*np.pi],
            door_marker_texture='materials/textures/marker6x6_2.png',
        )
        printer3.spawn_fast()

    else:
        node = make_webcam_node()

        printer2 = Simulated3DPrinter(
            node=node, pos=[0.40, -0.3, 0.065], orient=[0.0, 0.0, np.pi],
            door_marker_texture='materials/textures/marker6x6_1.png',
        )
        printer3 = Simulated3DPrinter(
            node=node, pos=[0.65, 0.1, 0.075], orient=[0.0, 0.0, 3/2*np.pi],
            door_marker_texture='materials/textures/marker6x6_2.png',
        )

    spin_in_background(node)
    wait_for_joint_states(node)

    node.get_logger().info("Starting initial scan for markers 1 and 2...")
    node.load_state()
    # NOTE: unlike runScrapePlate.py / runFullAutomationWithScrape.py, this script
    # does NOT lock the scrape marker. If you drive scrapePlate from the menu and
    # see the scrape approach drift / collide between repetitions, add
    #   node.lock_marker(<scrape_marker_id>)
    # here so the scrape marker stays pinned to its file pose. (Left off because
    # this is an interactive scan/debug tool, not a fixed scrape loop.)

    if runVirtual:
        bad_pos, bad_euler = printer2.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=1, bad_pos=bad_pos, bad_euler=bad_euler)
        bad_pos, bad_euler = printer3.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=2, bad_pos=bad_pos, bad_euler=bad_euler)

    else:
        viewing_distance = 0.15
        node.marker_offset_config[1] = 'box_offset'
        node.marker_offset_config[2] = 'printer_offset'

        bad_pos, bad_euler = printer2.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=1, bad_pos=bad_pos, bad_euler=bad_euler)
        bad_pos, bad_euler = printer3.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=2, bad_pos=bad_pos, bad_euler=bad_euler)

        node.register_printers([
            {"marker_id": 1, "pos": [0.48, -0.3, 0.07], "orient": [0.0, 0.0, np.pi],
             "door_marker_texture": 'materials/textures/marker6x6_1.png'},
            {"marker_id": 2, "pos": [0.65, 0.1, 0.07], "orient": [0.0, 0.0, 3/2*np.pi],
             "door_marker_texture": 'materials/textures/marker6x6_2.png'},
        ])

        node.scanMarkerApproach(marker_id=1, viewing_distance=viewing_distance)
        node.scanMarkerApproach(marker_id=2, viewing_distance=viewing_distance)

    node.get_logger().info("Initial scan complete.")
    run_command_menu(node)


if __name__ == '__main__':
    main()
