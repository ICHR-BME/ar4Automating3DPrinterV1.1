#!/usr/bin/env python3
"""
Pick the plate off one printer, scrape it against another, put it back.
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
RUN_SIM         = 0           # 1 = Gazebo (sim camera + spawned printers), 0 = hardware
ROBOT           = 'lite6'     # 'ar4' | 'lite6' | 'xarm6' (sim launch: launchVirtualRobot.sh / launchVirtualXArmLite6.sh / launchVirtualXArm6.sh)
SOURCE_ID       = 2           # marker to pick the plate from (and return it to)
SCRAPE_ID       = 1           # marker whose surface gets scraped against
# all motion and tilt angles come from the waypoint lists in
# printerAutomation.__init__'s offset_configs


def main():
    rclpy.init()
    node = start_node(sim=RUN_SIM, robot=ROBOT)
    speed_scale = 0.2
    node.moveit2.max_velocity = speed_scale
    node.moveit2.max_acceleration = speed_scale
    if RUN_SIM:
        # spawn printers instead of loading the save file, its real-world
        # poses don't match the sim scene
        spawn_sim_printers(node, sim_printer_specs(ROBOT, 2))
    else:
        # save file restores marker poses, offset config, printer configs.
        # Markers are pinned by default (updates only during their own scan
        # waypoints), so the scrape marker keeps its file pose automatically.
        if not node.load_state():
            node.get_logger().error("No save file found — run printer_automation.py first to create one.")
            return

        restore_saved_printers(node)

    # the post-scrape wrist roll is a {'rotate': deg} entry in the scrape
    # waypoint list (offset_configs), not an argument here
    ok = node.scrapePlate(
        source_id=SOURCE_ID,
        scrape_id=SCRAPE_ID,
        wait_after_pickup=False,
        wait_duration=10.0,
    )
    node.get_logger().info("scrapePlate succeeded." if ok else "scrapePlate failed.")


if __name__ == '__main__':
    main()
