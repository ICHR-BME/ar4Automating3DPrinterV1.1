#!/usr/bin/env python3
"""
Approach the plate on a printer and pick it up, then stop.
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
    require_scanned_markers,
)

# ---- Configuration ----
RUN_SIM         = 0           # 1 = Gazebo (sim camera + spawned printers), 0 = hardware
ROBOT           = 'lite6'     # 'ar4' | 'lite6' | 'xarm6' (sim launch: launchVirtualRobot.sh / launchVirtualXArmLite6.sh / launchVirtualXArm6.sh)
SOURCE_ID       = 2          # marker to pick the plate from
# all motion and tilt angles come from the waypoint lists in
# printerAutomation.__init__'s offset_configs


def main():
    rclpy.init()
    node = start_node(sim=RUN_SIM, robot=ROBOT)
    speed_scale = 0.15
    node.moveit2.max_velocity = speed_scale
    node.moveit2.max_acceleration = speed_scale
    if RUN_SIM:
        # spawn printers instead of loading the save file, its real-world
        # poses don't match the sim scene
        spawn_sim_printers(node, sim_printer_specs(ROBOT, 2))
    else:
        # save file restores marker poses, offset config, printer configs.
        # Markers are pinned by default — the source marker only updates
        # during its own pickup-scan windows, so moves can't drift it.
        if not node.load_state():
            node.get_logger().error("No save file found — run printer_automation.py first to create one.")
            return

        restore_saved_printers(node)
        # only officially scanned poses (scanFor2Markers.py) are accepted here
        # — hand-taught/geometric estimates must go through a scan first
        if not require_scanned_markers(node, [SOURCE_ID]):
            return

    ok = node.pickupOnly(
        source_id=SOURCE_ID,
        wait_after_pickup=False,
        wait_duration=10.0,
    )
    node.get_logger().info("pickupOnly succeeded." if ok else "pickupOnly failed.")


if __name__ == '__main__':
    main()
