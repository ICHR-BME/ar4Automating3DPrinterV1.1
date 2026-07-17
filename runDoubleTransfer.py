#!/usr/bin/env python3
"""
Automated double plate transfer.

Loads all configuration (printer positions, marker poses, offset config) from
the save file written by printer_automation.py, then runs transferPlate twice
per iteration:
  1. source=2, dest=0, rescan=1
  2. source=2, dest=1, rescan=0
(All motion, including the marker scans before each pickup, comes from the
waypoint lists in printerAutomation.__init__'s offset_configs.)

Set RUN_SIM = 1 below to run against Gazebo instead of hardware (start the sim
first with scripts/launchVirtualRobot.sh): the camera comes from the simulated
RGBD camera and three simulated printers are spawned in place of the save file.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy

from ar4_automation.runner_common import (
    start_node,
    restore_saved_printers,
    spawn_sim_printers,
    SIM_PRINTER_SPECS_3,
)


RUN_SIM = 1      # 1 = run in Gazebo (sim camera + spawned printers), 0 = hardware
NUM_REPEATS = 6  # Number of times to repeat the double transfer


def main():
    rclpy.init()
    node = start_node(sim=RUN_SIM)

    if RUN_SIM:
        # Spawn simulated printers (markers 0, 1, 2) instead of loading the
        # hardware save file — its real-world poses don't match the sim scene.
        spawn_sim_printers(node, SIM_PRINTER_SPECS_3)
    else:
        # Load save file — restores marker poses, offset config, and printer configs
        if not node.load_state():
            node.get_logger().error("No save file found — run printer_automation.py first to create one.")
            return

        restore_saved_printers(node)
    
    for i in range(NUM_REPEATS):
        node.get_logger().info(f"=== Iteration {i + 1}/{NUM_REPEATS} ===")

        # Transfer 1: source=2, dest=0, rescan=1
        ok1 = node.transferPlate(source_id=2, dest_id=0, rescan_id=1)

        # Transfer 2: source=2, dest=1, rescan=0
        ok2 = node.transferPlate(source_id=2, dest_id=1, rescan_id=0)

    node.get_logger().info("All transfers complete.")
    node.save_state()


if __name__ == '__main__':
    main()
