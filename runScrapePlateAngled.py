#!/usr/bin/env python3
"""
Automated scrape plate run with configurable tool tilt angles.

Same sequence as runScrapePlate.py, but the tool orientation can be tilted by
a configurable angle (degrees, about the marker's local X axis) independently
for three phases:
  PICKUP_ANGLE_DEG — while approaching, grasping, and lifting the build plate
  SCRAPE_ANGLE_DEG — during the scrape approach, full depth, and retract
  PLACE_ANGLE_DEG  — while lowering and releasing the plate back at the source

Notes:
  - 0.0 for all three angles reproduces runScrapePlate.py exactly.
  - If PLACE_ANGLE_DEG == PICKUP_ANGLE_DEG the placement replays the recorded
    pickup joints (wrist-continuous). A different place angle forces pose-based
    placement, where MoveIt's IK picks the wrist configuration itself.

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
RUN_SIM         = 1           # 1 = run in Gazebo (sim camera + spawned printers), 0 = hardware
SOURCE_ID       = 2           # Marker to pick up the plate from (and return it to)
SCRAPE_ID       = 1           # Marker whose surface the plate is scraped against
SCAN_DISTANCE   = 0.15        # Distance (m) used when scanning markers
SCRAPE_STANDOFF = 0.38        # Distance (m) along scrape marker Z to approach from

# Tool tilt angles (degrees, about the marker's local X axis; 0.0 = no tilt)
PICKUP_ANGLE_DEG = 5.0        # Tilt while approaching/grasping/lifting the plate
SCRAPE_ANGLE_DEG = 10.0       # Tilt during scrape approach, depth, and retract
PLACE_ANGLE_DEG  = 5.0        # Tilt while lowering/releasing the plate


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
        scan_distance=SCAN_DISTANCE,
        scrape_standoff=SCRAPE_STANDOFF,
        wait_after_pickup=False,
        wait_duration=10.0,
        rotate_after_scrape=True,
        pickup_angle_deg=PICKUP_ANGLE_DEG,
        place_angle_deg=PLACE_ANGLE_DEG,
        scrape_angle_deg=SCRAPE_ANGLE_DEG,
    )
    node.get_logger().info("scrapePlate succeeded." if ok else "scrapePlate failed.")


if __name__ == '__main__':
    main()
