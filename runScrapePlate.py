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
RUN_SIM         = 1           # 1 = Gazebo (sim camera + spawned printers), 0 = hardware
ROBOT           = 'lite6'       # 'ar4' or 'lite6' (sim launch: launchVirtualRobot.sh / launchVirtualXArmLite6.sh)
SOURCE_ID       = 2           # marker to pick the plate from (and return it to)
SCRAPE_ID       = 1           # marker whose surface gets scraped against
# all motion and tilt angles come from the waypoint lists in
# printerAutomation.__init__'s offset_configs


def main():
    rclpy.init()
    node = start_node(sim=RUN_SIM, robot=ROBOT)

    if RUN_SIM:
        # spawn printers instead of loading the save file, its real-world
        # poses don't match the sim scene
        spawn_sim_printers(node, sim_printer_specs(ROBOT, 2))
    else:
        # save file restores marker poses, offset config, printer configs
        if not node.load_state():
            node.get_logger().error("No save file found — run printer_automation.py first to create one.")
            return

        # pin the scrape marker to its file pose so live detections can't
        # drift it between repetitions
        node.lock_marker(SCRAPE_ID)

        restore_saved_printers(node)

    ok = node.scrapePlate(
        source_id=SOURCE_ID,
        scrape_id=SCRAPE_ID,
        wait_after_pickup=False,
        wait_duration=10.0,
        rotate_after_scrape=True,
    )
    node.get_logger().info("scrapePlate succeeded." if ok else "scrapePlate failed.")


if __name__ == '__main__':
    main()
