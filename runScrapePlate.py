#!/usr/bin/env python3
"""
Automated scrape plate run.

Loads all configuration (printer positions, marker poses, offset config) from
the save file written by printer_automation.py, then runs scrapePlate once:
  source=SOURCE_ID  — marker to pick up the plate from and return it to
  scrape=SCRAPE_ID  — marker whose surface the plate is scraped against

Set RUN_SIM = 1 below to run against Gazebo instead of hardware (start the sim
first with scripts/launchVirtualRobot.sh): the camera comes from the simulated
RGBD camera and two simulated printers are spawned in place of the save file.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy

from ar4_automation.runner_common import (
    start_node,
    restore_saved_printers,
    spawn_sim_printers,
    SIM_PRINTER_SPECS_2,
)


# ---- Configuration ----
RUN_SIM         = 1           # 1 = run in Gazebo (sim camera + spawned printers), 0 = hardware
SOURCE_ID       = 2           # Marker to pick up the plate from (and return it to)
SCRAPE_ID       = 1           # Marker whose surface the plate is scraped against
SCAN_DISTANCE   = 0.15         # Distance (m) used when scanning markers
SCRAPE_STANDOFF = 0.38         # Distance (m) along scrape marker Z to approach from
# Scrape contact offset is set via node.scrape_offset (defined in printerAutomation.__init__)


def main():
    rclpy.init()
    node = start_node(sim=RUN_SIM)

    if RUN_SIM:
        # Spawn simulated printers (markers 1 and 2) instead of loading the
        # hardware save file — its real-world poses don't match the sim scene.
        spawn_sim_printers(node, SIM_PRINTER_SPECS_2)
    else:
        # Load save file — restores marker poses, offset config, and printer configs
        if not node.load_state():
            node.get_logger().error("No save file found — run printer_automation.py first to create one.")
            return

        # The scrape marker is a fixed reference: pin it to the file-loaded pose so the
        # camera can never overwrite it. This keeps the scrape approach identical on
        # every repetition (without it, live detections drift marker 1 between passes).
        node.lock_marker(SCRAPE_ID)

        restore_saved_printers(node)

    for i in range(1):
        ok = node.scrapePlate(
            source_id=SOURCE_ID,
            scrape_id=SCRAPE_ID,
            scan_distance=SCAN_DISTANCE,
            scrape_standoff=SCRAPE_STANDOFF,
            wait_after_pickup=False,
            wait_duration=10.0,
            rotate_after_scrape=True,
        )
        node.get_logger().info("scrapePlate succeeded." if ok else "scrapePlate failed.")

        #node.save_state()


if __name__ == '__main__':
    main()
