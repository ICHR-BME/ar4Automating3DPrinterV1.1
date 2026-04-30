#!/usr/bin/env python3
"""
Automated scrape plate run.

Loads all configuration (printer positions, marker poses, offset config) from
the save file written by 3DPrinterAutomation.py, then runs scrapePlate once:
  source=SOURCE_ID  — marker to pick up the plate from and return it to
  scrape=SCRAPE_ID  — marker whose surface the plate is scraped against
"""

import sys
import os
import time
import threading
import importlib
import numpy as np
import rclpy

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

# The source file starts with a digit, so standard import syntax won't work.
_mod = importlib.util.spec_from_file_location(
    "ThreeDPrinterAutomation",
    os.path.join(_SCRIPT_DIR, "3DPrinterAutomation.py"),
)
_automation = importlib.util.module_from_spec(_mod)
_mod.loader.exec_module(_automation)
printerAutomation = _automation.printerAutomation

from simulated3DPrinter import Simulated3DPrinter


# ---- Configuration ----
SOURCE_ID       = 2           # Marker to pick up the plate from (and return it to)
SCRAPE_ID       = 1           # Marker whose surface the plate is scraped against
SCAN_DISTANCE   = 0.15         # Distance (m) used when scanning markers
SCRAPE_STANDOFF = 0.38         # Distance (m) along scrape marker Z to approach from
# Scrape contact offset is set via node.scrape_offset (defined in printerAutomation.__init__)



def main():
    rclpy.init()

    node = printerAutomation(
        calibration_mode=False,
        stream_source="webcam",
        feed_rotation_deg=90.0,
        marker_sizes=[0.03, 0.025],
    )
    node.stream.distance_scale = 1.0 / 0.702

    # Executor
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    if hasattr(node.stream, "_ros_node"):
        executor.add_node(node.stream._ros_node)
    else:
        stream_thread = threading.Thread(target=node.stream.run, daemon=True)
        stream_thread.start()

    def _resilient_spin(executor):
        while rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                node.get_logger().warn(f"Executor spin error (recovering): {e}")
                time.sleep(0.01)

    spin_thread = threading.Thread(target=_resilient_spin, args=(executor,), daemon=True)
    spin_thread.start()

    # Wait for joint_states
    node.get_logger().info("Waiting for joint_states...")
    for _ in range(100):
        if node._last_joint_msg is not None:
            break
        time.sleep(0.1)
    else:
        node.get_logger().warn("Timed out waiting for joint_states — proceeding anyway")

    # Load save file — restores marker poses, offset config, and printer configs
    if not node.load_state():
        node.get_logger().error("No save file found — run 3DPrinterAutomation.py first to create one.")
        return

    # Reconstruct Simulated3DPrinter objects from the saved printer configs so
    # register_estimated_marker has the correct geometric estimates as a fallback.
    for p in getattr(node, '_saved_printer_configs', []):
        printer = Simulated3DPrinter(
            node=node,
            pos=p["pos"],
            orient=p["orient"],
            door_marker_texture=p["door_marker_texture"],
        )
        bad_pos, bad_euler = printer.get_door_marker_pose_in_base()
        existing = node._find_marker_entry(p["marker_id"])
        if existing is None or existing.get("estimated"):
            node.register_estimated_marker(
                marker_id=p["marker_id"], bad_pos=bad_pos, bad_euler=bad_euler
            )

    ok = node.scrapePlate(
        source_id=SOURCE_ID,
        scrape_id=SCRAPE_ID,
        scan_distance=SCAN_DISTANCE,
        scrape_standoff=SCRAPE_STANDOFF,
        wait_after_pickup=True,
        wait_duration=10.0,
        rotate_after_scrape=True,
    )
    node.get_logger().info("scrapePlate succeeded." if ok else "scrapePlate failed.")

    node.save_state()


if __name__ == '__main__':
    main()
