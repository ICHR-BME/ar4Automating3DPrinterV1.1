#!/usr/bin/env python3
"""
Automated scrape plate run with tilted tool orientations.

Same sequence as runScrapePlate.py.  The tool tilt angles (about the marker's
local X axis) are no longer set here: each waypoint carries its own angle_deg
in the offset_configs waypoint lists (printerAutomation.__init__) — the
pickup/place lists of the source marker's config and the scrape list of the
scrape marker's config.

Note: a config whose place list has its own descent waypoints uses pose-based
placement (MoveIt's IK picks the wrist configuration) instead of replaying the
recorded pickup joints.

Set RUN_SIM = 1 below to run against Gazebo instead of hardware (start the sim
first with scripts/launchVirtualRobot.sh).
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
RUN_SIM         = 0           # 1 = run in Gazebo (sim camera + spawned printers), 0 = hardware
SOURCE_ID       = 2           # Marker to pick up the plate from (and return it to)
SCRAPE_ID       = 1           # Marker whose surface the plate is scraped against
# All motion (scans, approach/grasp/carry, scrape standoff/depth/retract,
# placement and withdraw) and all tool tilt angles come from the waypoint
# lists in printerAutomation.__init__'s offset_configs.


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

        # Pin the scrape marker to its file-loaded pose so live detections can't
        # drift it between repetitions (see runScrapePlate.py).
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
