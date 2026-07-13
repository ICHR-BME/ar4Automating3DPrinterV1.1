#!/usr/bin/env python3
"""
Print-and-scrape loop.

The prints to run come from a YAML queue file (PRINT_QUEUE_FILE, default
print_queue.yaml) that lists the print folder, the file names and how many
times to print each one. For every print in the queue:
  1. Start the print on the Bambu printer associated with SOURCE_ID.
  2. Wait for the print to finish.
  3. Move the printer head out of the way (prepare_for_pickup).
  4. Scrape the plate (pick up from SOURCE_ID, scrape against SCRAPE_ID, return).

Edit the ---- Configuration ---- section and the queue file before running.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy
import yaml

from runner_common import start_webcam_node, restore_saved_printers
from simulated3DPrinter import Simulated3DPrinter
from printerclass import BambuPrinter, load_printer_config, strip_startup_gcode


# ---- Configuration ----
SOURCE_ID       = 2           # Marker to pick up the plate from (and return it to)
SCRAPE_ID       = 1           # Marker whose surface the plate is scraped against
SCAN_DISTANCE   = 0.15        # Distance (m) used when scanning markers
SCRAPE_STANDOFF = 0.38        # Distance (m) along scrape marker Z to approach from

# Bambu printer to use. Credentials are loaded from printer_config.yaml
# (copy printer_config.example.yaml and fill it in). The previously hard-coded
# values here matched the 'a1_mini_2' entry.
PRINTER_NAME = "a1_mini"

# YAML file describing the prints to run: the print folder, the file names
# and how many times to print each one (see print_queue.yaml).
PRINT_QUEUE_FILE = "print_queue.yaml"

# Remove the printer's startup procedures (sound, purge, mech-mode check,
# nozzle wipe on the plate, bed leveling, re-homing) from the print file
# before uploading. Keeps only a heat-and-home preamble plus the nozzle-load
# blob squirt, so the print starts directly. The startup shoves the plate
# around and breaks the scrape automation.
STRIP_STARTUP = True
# ---- End Configuration ----


def load_print_queue(queue_file):
    """
    Reads the YAML print queue and returns the flat, ordered list of local
    print-file paths to run — each file repeated 'count' times.

    Expected format:
        print_folder: testPrints
        prints:
          - name: bed_scraper_a1mini.gcode.3mf
            count: 20
          - name: BenchyFast.3mf
            count: 1
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    queue_path = os.path.join(base_dir, queue_file)
    with open(queue_path, "r") as f:
        cfg = yaml.safe_load(f)

    folder = cfg.get("print_folder", "")
    entries = cfg.get("prints") or []
    if not entries:
        raise ValueError(f"No 'prints' entries found in {queue_path}")

    queue = []
    for entry in entries:
        name = entry["name"]
        count = int(entry.get("count", 1))
        if count < 1:
            continue
        local_file = os.path.join(folder, name)
        if not os.path.exists(os.path.join(base_dir, local_file)):
            raise FileNotFoundError(f"Print file not found: {local_file} "
                                    f"(from {queue_path})")
        queue.extend([local_file] * count)
    return queue


def main():
    # Load and validate the queue first so a bad YAML fails before any
    # hardware is touched.
    print_queue = load_print_queue(PRINT_QUEUE_FILE)

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

    # Prepare and upload each unique file in the queue once, and remember the
    # name to print it under. upload_file_timeout stores files at the SD card
    # root under their basename, so prints are started by basename.
    remote_names = {}
    for local_file in dict.fromkeys(print_queue):  # unique, in queue order
        if STRIP_STARTUP:
            try:
                local_file_prepped = strip_startup_gcode(local_file)
                node.get_logger().info(
                    f"Stripped startup gcode: {local_file} -> {local_file_prepped}"
                )
            except ValueError as e:
                # e.g. a file sliced with a custom start-gcode profile that
                # doesn't have the stock Bambu markers — print it as-is.
                node.get_logger().warning(
                    f"Could not strip startup from {local_file}: {e} "
                    f"Uploading it UNMODIFIED (full startup will run)."
                )
                local_file_prepped = local_file
        else:
            local_file_prepped = local_file
        if not bambu.upload_file_timeout(local_file_prepped):
            raise RuntimeError(f"Upload failed for {local_file_prepped}")
        remote_names[local_file] = os.path.basename(local_file_prepped)

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

    # Main print-and-scrape loop over the queue
    total = len(print_queue)
    for cycle, local_file in enumerate(print_queue, start=1):
        remote_file = remote_names[local_file]
        node.get_logger().info(
            f"=== Cycle {cycle}/{total}: starting print {remote_file} ==="
        )

        bambu.start_print(remote_file)
        bambu.waitUntilPrintFinished()

        node.get_logger().info(f"=== Cycle {cycle}/{total}: print done, preparing for pickup ===")
        bambu.prepare_for_pickup()

        node.get_logger().info(f"=== Cycle {cycle}/{total}: scraping plate ===")
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
            f"=== Cycle {cycle}/{total}: scrapePlate {'succeeded' if ok else 'FAILED'} ==="
        )

        #node.save_state()

    node.get_logger().info("All cycles complete.")
    bambu.disconnect()


if __name__ == '__main__':
    main()
