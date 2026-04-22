#!/usr/bin/env python3
"""
Automated double plate transfer.

Loads all configuration (printer positions, marker poses, offset config) from
the save file written by 3DPrinterAutomation.py, then runs transferPlate twice:
  1. source=2, dest=0, rescan=1  (scan_distance=0.15)
  2. source=2, dest=1, rescan=0  (scan_distance=0.15)
"""

import sys
import os
import time
import threading
import importlib
import json
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
        # Only register if this marker wasn't already restored from the save
        # (load_state restores real detections; this provides a geometric fallback).
        existing = node._find_marker_entry(p["marker_id"])
        if existing is None or existing.get("estimated"):
            node.register_estimated_marker(
                marker_id=p["marker_id"], bad_pos=bad_pos, bad_euler=bad_euler
            )

    for i in range(NUM_REPEATS):
        node.get_logger().info(f"=== Iteration {i + 1}/{NUM_REPEATS} ===")

        # Transfer 1: source=2, dest=0, rescan=1
        ok1 = node.transferPlate(source_id=2, dest_id=0, rescan_id=1, scan_distance=0.15)
        

        # Transfer 2: source=2, dest=1, rescan=0
        #ok2 = node.transferPlate(source_id=2, dest_id=1, rescan_id=0, scan_distance=0.15)
        
        #ok2 = node.transferPlate(source_id=1, dest_id=2, rescan_id=2, scan_distance=0.15)
        #node.get_logger().info("Transfer 2 succeeded." if ok2 else "Transfer 2 failed.")



    node.get_logger().info("All transfers complete.")
    node.save_state()


NUM_REPEATS = 1  # Number of times to repeat the double transfer

if __name__ == '__main__':
    main()

