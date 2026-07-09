#!/usr/bin/env python3
"""
Automated double plate transfer.

Loads all configuration (printer positions, marker poses, offset config) from
the save file written by printer_automation.py, then runs transferPlate twice
per iteration:
  1. source=2, dest=0, rescan=1  (scan_distance=0.15)
  2. source=2, dest=1, rescan=0  (scan_distance=0.15)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy

from runner_common import start_webcam_node, restore_saved_printers


NUM_REPEATS = 6  # Number of times to repeat the double transfer


def main():
    rclpy.init()
    node = start_webcam_node()

    # Load save file — restores marker poses, offset config, and printer configs
    if not node.load_state():
        node.get_logger().error("No save file found — run printer_automation.py first to create one.")
        return

    restore_saved_printers(node)

    for i in range(NUM_REPEATS):
        node.get_logger().info(f"=== Iteration {i + 1}/{NUM_REPEATS} ===")

        # Transfer 1: source=2, dest=0, rescan=1
        ok1 = node.transferPlate(source_id=2, dest_id=0, rescan_id=1, scan_distance=0.15)

        # Transfer 2: source=2, dest=1, rescan=0
        ok2 = node.transferPlate(source_id=2, dest_id=1, rescan_id=0, scan_distance=0.15)

    node.get_logger().info("All transfers complete.")
    node.save_state()


if __name__ == '__main__':
    main()
