#!/usr/bin/env python3
"""
Double plate transfer loop: transferPlate 2->0 then 2->1 each iteration,
config from printer_automation.py's save file. All motion comes from the
offset_configs waypoint lists in printerAutomation.__init__.
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


RUN_SIM = 1      # 1 = Gazebo (sim camera + spawned printers), 0 = hardware
ROBOT = 'xarm6'  # 'ar4' | 'lite6' | 'xarm6' (sim launch: launchVirtualRobot.sh / launchVirtualXArmLite6.sh / launchVirtualXArm6.sh)
NUM_REPEATS = 6  # times to repeat the double transfer


def main():
    rclpy.init()
    node = start_node(sim=RUN_SIM, robot=ROBOT)

    if RUN_SIM:
        # spawn printers instead of loading the save file, its real-world
        # poses don't match the sim scene
        spawn_sim_printers(node, sim_printer_specs(ROBOT, 3))
    else:
        # save file restores marker poses, offset config, printer configs
        if not node.load_state():
            node.get_logger().error("No save file found — run printer_automation.py first to create one.")
            return

        restore_saved_printers(node)
    
    for i in range(NUM_REPEATS):
        node.get_logger().info(f"=== Iteration {i + 1}/{NUM_REPEATS} ===")

        ok1 = node.transferPlate(source_id=2, dest_id=0, rescan_id=1)

        ok2 = node.transferPlate(source_id=2, dest_id=1, rescan_id=0)

    node.get_logger().info("All transfers complete.")
    node.save_state()


if __name__ == '__main__':
    main()
