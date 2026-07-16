#!/usr/bin/env python3
"""
Shared boilerplate for the runner/entry scripts (runScrapePlate.py,
runDoubleTransfer.py, runFullAutomationWithScrape.py, scanFor2Markers.py,
scanFor3Markers.py):

  - building a printerAutomation node with the standard webcam configuration
  - spinning it on a background MultiThreadedExecutor
  - waiting for joint_states
  - reconstructing Simulated3DPrinter estimates from a loaded save file
  - the interactive command menu used by the scanForNMarkers scripts
"""

import time
import threading

import rclpy

from .printer_automation import printerAutomation
from .simulated3DPrinter import Simulated3DPrinter

# Standard hardware configuration shared by all runner scripts.
WEBCAM_NODE_KWARGS = dict(
    calibration_mode=False,
    stream_source="webcam",
    feed_rotation_deg=90.0,
    marker_sizes=[0.03, 0.025],
)
WEBCAM_DISTANCE_SCALE = 1.0 / 0.702


def make_webcam_node(**overrides):
    """Build a printerAutomation node with the standard webcam config."""
    kwargs = dict(WEBCAM_NODE_KWARGS)
    kwargs.update(overrides)
    node = printerAutomation(**kwargs)
    node.stream.distance_scale = WEBCAM_DISTANCE_SCALE
    return node


def spin_in_background(node):
    """Start a MultiThreadedExecutor for the node (and its stream) in a daemon thread."""
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    if hasattr(node.stream, "_ros_node"):
        executor.add_node(node.stream._ros_node)
    else:
        threading.Thread(target=node.stream.run, daemon=True).start()

    def _resilient_spin():
        while rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                node.get_logger().warn(f"Executor spin error (recovering): {e}")
                time.sleep(0.01)

    threading.Thread(target=_resilient_spin, daemon=True).start()
    return executor


def wait_for_joint_states(node, timeout=10.0):
    """Block until the first joint_states message arrives (or warn on timeout)."""
    node.get_logger().info("Waiting for joint_states...")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if node._last_joint_msg is not None:
            return True
        time.sleep(0.1)
    node.get_logger().warn("Timed out waiting for joint_states — proceeding anyway")
    return False


def start_webcam_node(**overrides):
    """make_webcam_node + spin_in_background + wait_for_joint_states."""
    node = make_webcam_node(**overrides)
    spin_in_background(node)
    wait_for_joint_states(node)
    return node


def restore_saved_printers(node):
    """
    Reconstruct Simulated3DPrinter objects from the printer configs restored by
    node.load_state(), and register their geometric door-marker estimates as a
    fallback for any marker that has no real (non-estimated) saved pose.
    """
    for p in getattr(node, "_saved_printer_configs", []):
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


# ---------------------------------------------------------------------------
# Interactive command menu (used by scanFor2Markers.py / scanFor3Markers.py)
# ---------------------------------------------------------------------------

def _print_menu():
    print("\n" + "=" * 50)
    print("  3D Printer Automation - Command Menu")
    print("=" * 50)
    print("  1) Scan location for markers (manual pos/orient)")
    print("  2) Move to marker")
    print("  3) Pickup plate (move + lift)")
    print("  4) Place plate at marker")
    print("  5) List detected markers")
    print("  6) Scan to marker (by ID, uses TF)")
    print("  7) Go home & resync (correct step-loss drift)")
    print("  8) Transfer plate (source → dest, rescan → place)")
    print("  9) Scrape plate (pickup → scrape surface → return)")
    print("=" * 50)


def _parse_floats(prompt, count=None):
    """Prompt for space-separated floats. Returns list of floats or None on error."""
    try:
        raw = input(prompt).strip()
        values = [float(v) for v in raw.split()]
        if count is not None and len(values) != count:
            print(f"  Expected {count} values, got {len(values)}")
            return None
        return values
    except ValueError:
        print("  Invalid input. Enter space-separated numbers.")
        return None


def run_command_menu(node):
    """
    Read user commands from stdin and dispatch them on the node (which must
    already be spinning in a background thread). Blocks until EOF/shutdown.
    """
    time.sleep(5.0)
    print("\n[INFO] System ready. Type a command number.")
    node.record_startup_time()

    while rclpy.ok():
        _print_menu()
        try:
            choice = input(">> ").strip()
        except EOFError:
            break

        # ROS log messages can interleave with terminal input, causing extra
        # characters to be buffered before the intended option digit(s).
        _valid_choices = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
        if choice not in _valid_choices and len(choice) >= 2 and choice[1:] in _valid_choices:
            choice = choice[1:]

        if choice == "1":
            pos = _parse_floats("  Enter estimated pos (x y z): ", 3)
            if pos is None:
                continue
            orient = _parse_floats("  Enter estimated orient (roll pitch yaw) [0 0 0]: ", 3)
            if orient is None:
                orient = [0.0, 0.0, 0.0]
            dist = _parse_floats("  Viewing distance [0.15]: ", 1)
            dist = dist[0] if dist else 0.15
            node.get_logger().info(f"User requested scanLocationForMarkers at {pos}")
            node.scanLocationForMarkers(estimated_pos=pos, estimated_orient=orient, viewing_distance=dist)

        elif choice == "2":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            node.get_logger().info(f"User requested moveToMarker({mid})")
            node.moveToMarker(markerID=mid)

        elif choice == "3":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            node.get_logger().info(f"User requested pickupPlate({mid})")
            node.pickupPlate(markerID=mid)

        elif choice == "4":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            node.get_logger().info(f"User requested placePlate({mid})")
            node.placePlate(markerID=mid)

        elif choice == "5":
            markers = node.marker_poses
            if markers:
                print(f"\n  Found {len(markers)} marker(s):")
                for entry in markers:
                    pos = entry.get('positionInWorld', 'N/A')
                    ori = entry.get('orientInWorld', 'N/A')
                    print(f"    ID {entry['id']}: pos={pos}, orient={ori}")
            else:
                print("  No markers found yet.")

        elif choice == "6":
            mid = _parse_floats("  Marker ID [0]: ", 1)
            mid = int(mid[0]) if mid else 0
            dist = _parse_floats("  Viewing distance [0.15]: ", 1)
            dist = dist[0] if dist else 0.15
            node.get_logger().info(f"User requested scanToMarker({mid}, dist={dist})")
            node.scanToMarker(marker_id=mid, viewing_distance=dist)

        elif choice == "7":
            scale = _parse_floats("  Velocity scaling [0.2]: ", 1)
            scale = scale[0] if scale else 0.2
            node.get_logger().info(f"User requested go_home(velocity_scaling={scale})")
            node.go_home(velocity_scaling=scale)

        elif choice == "8":
            ids = _parse_floats("  Source, dest, rescan marker IDs (e.g. 1 2 1): ", 3)
            if ids is None:
                continue
            source_id, dest_id, rescan_id = int(ids[0]), int(ids[1]), int(ids[2])
            dist = _parse_floats("  Scan distance [0.15]: ", 1)
            dist = dist[0] if dist else 0.15
            node.get_logger().info(
                f"User requested transferPlate({source_id}, {dest_id}, {rescan_id}, scan_distance={dist})"
            )
            node.transferPlate(source_id=source_id, dest_id=dest_id, rescan_id=rescan_id, scan_distance=dist)

        elif choice == "9":
            ids = _parse_floats("  Source, scrape marker IDs (e.g. 1 2): ", 2)
            if ids is None:
                continue
            source_id, scrape_id = int(ids[0]), int(ids[1])
            dist = _parse_floats("  Scan distance [0.15]: ", 1)
            dist = dist[0] if dist else 0.15
            standoff = _parse_floats("  Scrape standoff distance [0.15]: ", 1)
            standoff = standoff[0] if standoff else 0.15
            offset = _parse_floats("  Scrape offset in marker frame (x y z) [0 0 0.05]: ", 3)
            offset = offset if offset else None
            node.get_logger().info(
                f"User requested scrapePlate({source_id}, {scrape_id}, scan_distance={dist}, "
                f"scrape_standoff={standoff}, scrape_offset={offset})"
            )
            node.scrapePlate(
                source_id=source_id, scrape_id=scrape_id,
                scan_distance=dist, scrape_standoff=standoff, scrape_offset=offset,
            )

        else:
            print("  Unknown option. Try again.")
