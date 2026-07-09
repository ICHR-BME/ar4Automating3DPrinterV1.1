#!/usr/bin/env python3
"""
Print-and-scrape loop.

Each iteration:
  1. Start a print on the Bambu printer associated with SOURCE_ID.
  2. Wait for the print to finish.
  3. Move the printer head out of the way (prepare_for_pickup).
  4. Scrape the plate (pick up from SOURCE_ID, scrape against SCRAPE_ID, return).
  5. Repeat for NUM_CYCLES iterations.

Edit the ---- Configuration ---- section before running.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy

from runner_common import start_webcam_node, restore_saved_printers
from simulated3DPrinter import Simulated3DPrinter
from printerclass import BambuPrinter, load_printer_config


# ---- Configuration ----
SOURCE_ID       = 2           # Marker to pick up the plate from (and return it to)
SCRAPE_ID       = 1           # Marker whose surface the plate is scraped against
SCAN_DISTANCE   = 0.15        # Distance (m) used when scanning markers
SCRAPE_STANDOFF = 0.38        # Distance (m) along scrape marker Z to approach from
NUM_CYCLES      = 20           # Number of print-then-scrape cycles to run

# Bambu printer to use. Credentials are loaded from printer_config.yaml
# (copy printer_config.example.yaml and fill it in). The previously hard-coded
# values here matched the 'a1_mini_2' entry.
PRINTER_NAME = "a1_mini"

# File name on the printer's SD card to print each cycle.
PRINT_FILENAME = "testPrints/bed_scraper_a1mini.gcode.3mf"

#PRINT_FILENAME = "testPrints/BenchyFast.3mf"
#PRINT_FILENAME = "testPrints/smallCylinderPLA15m17s.gcode.3mf"
# ---- End Configuration ----


def main():
    rclpy.init()
    node = start_webcam_node()

    # Load save file — restores marker poses, offset config, and printer configs
    if not node.load_state():
        node.get_logger().error(
            "No save file found — run printer_automation.py first to create one."
        )
        return

    # The scrape marker is a fixed reference: pin it to the file-loaded pose so
    # neither the camera nor the geometry re-registration / scan below can change
    # it. This keeps the scrape approach identical on every cycle and avoids the
    # marker-drift collisions fixed in runScrapePlate.py. (lock_marker is honoured
    # by both the camera update path and register_estimated_marker.)
    node.lock_marker(SCRAPE_ID)

    restore_saved_printers(node)

    # Connect the Bambu printer and register it with the node.
    printer_cfg = load_printer_config(PRINTER_NAME)
    bambu = BambuPrinter(
        ip=printer_cfg["ip"],
        access_code=printer_cfg["access_code"],
        serial=printer_cfg["serial"],
    )
    bambu.connect()
    bambu.enable_debug_listener()
    node.register_bambu_printer(SOURCE_ID, bambu)
    bambu.upload_file_timeout(PRINT_FILENAME)

    # Set up markers (mirrors scanFor2Markers.py non-virtual procedure). Marker 1
    # (scrape) is locked to the file pose above, so it is NOT estimated or scanned
    # here — those calls would be no-ops. Only marker 2 (the pickup source) is
    # estimated and scanned, since it is re-detected fresh each cycle.
    viewing_distance = SCAN_DISTANCE
    node.marker_offset_config[1] = 'box_offset'
    node.marker_offset_config[2] = 'printer_offset'

    printer3 = Simulated3DPrinter(
        node=node, pos=[0.65, 0.1, 0.075], orient=[0.0, 0.0, 3/2*np.pi],
        door_marker_texture='materials/textures/marker6x6_2.png',
    )
    bad_pos, bad_euler = printer3.get_door_marker_pose_in_base()
    node.register_estimated_marker(marker_id=2, bad_pos=bad_pos, bad_euler=bad_euler)

    node.register_printers([
        {"marker_id": 1, "pos": [0.48, -0.3, 0.07], "orient": [0.0, 0.0, np.pi],
         "door_marker_texture": 'materials/textures/marker6x6_1.png'},
        {"marker_id": 2, "pos": [0.65, 0.1, 0.07], "orient": [0.0, 0.0, 3/2*np.pi],
         "door_marker_texture": 'materials/textures/marker6x6_2.png'},
    ])

    node.get_logger().info("Scanning for source marker 2...")
    node.scanMarkerApproach(marker_id=2, viewing_distance=viewing_distance)
    node.get_logger().info("Initial scan complete.")

    # Main print-and-scrape loop
    for cycle in range(1, NUM_CYCLES + 1):
        node.get_logger().info(f"=== Cycle {cycle}/{NUM_CYCLES}: starting print ===")

        bambu.start_print(PRINT_FILENAME)
        bambu.waitUntilPrintFinished()

        node.get_logger().info(f"=== Cycle {cycle}/{NUM_CYCLES}: print done, preparing for pickup ===")
        bambu.prepare_for_pickup()

        node.get_logger().info(f"=== Cycle {cycle}/{NUM_CYCLES}: scraping plate ===")
        ok = node.scrapePlate(
            source_id=SOURCE_ID,
            scrape_id=SCRAPE_ID,
            scan_distance=SCAN_DISTANCE,
            scrape_standoff=SCRAPE_STANDOFF,
            wait_after_pickup=True,
            wait_duration=90.0,
            rotate_after_scrape=True,
        )
        node.get_logger().info(
            f"=== Cycle {cycle}/{NUM_CYCLES}: scrapePlate {'succeeded' if ok else 'FAILED'} ==="
        )

        #node.save_state()

    node.get_logger().info("All cycles complete.")
    bambu.disconnect()


if __name__ == '__main__':
    main()
