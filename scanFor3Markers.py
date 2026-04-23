#!/usr/bin/env python3
"""
Scan for 3 markers (IDs 0, 1, 2) then start the interactive command menu.
"""

import sys
import os
import importlib
import rclpy
import numpy as np
import time
import threading

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


def _print_menu():
    """Print the interactive command menu."""
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


def _input_thread(node):
    """
    Runs in a background thread. Reads user commands from stdin
    and dispatches them on the node (which is spinning in the main thread).
    """
    # Wait for the system to initialize
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
            ids = _parse_floats("  Source, dest, rescan marker IDs (e.g. 0 1 2): ", 3)
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
            ids = _parse_floats("  Source, scrape marker IDs (e.g. 0 1): ", 2)
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


def main():
    rclpy.init()
    runVirtual = 0

    if runVirtual:
        stream_source = "ros"
        node = printerAutomation(calibration_mode=False, stream_source=stream_source)
        node.gripper_disabled = True
        node.randomize_estimated_markers = True

        printer1 = Simulated3DPrinter(
            node=node, pos=[0.22, -0.2, 0.21], orient=[0.0, 0.0, np.pi],
            door_marker_texture='materials/textures/marker6x6_0.png',
        )
        printer1.spawn_fast()

        printer2 = Simulated3DPrinter(
            node=node, pos=[0.44, -0.2, 0.21], orient=[0.0, 0.1, np.pi],
            door_marker_texture='materials/textures/marker6x6_1.png',
        )
        printer2.spawn_fast()

        printer3 = Simulated3DPrinter(
            node=node, pos=[0.60, 0.1, 0.21], orient=[0.0, 0.0, 3/2*np.pi],
            door_marker_texture='materials/textures/marker6x6_2.png',
        )
        printer3.spawn_fast()

    else:
        stream_source = "webcam"
        node = printerAutomation(
            calibration_mode=False, stream_source=stream_source,
            feed_rotation_deg=90.0, marker_sizes=[0.03, 0.025],
        )
        node.stream.distance_scale = 1.0 / 0.702

        printer1 = Simulated3DPrinter(
            node=node, pos=[0.28, -0.3, 0.065], orient=[0.0, 0.0, np.pi],
            door_marker_texture='materials/textures/marker6x6_0.png',
        )
        printer2 = Simulated3DPrinter(
            node=node, pos=[0.50, -0.3, 0.065], orient=[0.0, 0.0, np.pi],
            door_marker_texture='materials/textures/marker6x6_1.png',
        )
        printer3 = Simulated3DPrinter(
            node=node, pos=[0.65, 0.1, 0.075], orient=[0.0, 0.0, 3/2*np.pi],
            door_marker_texture='materials/textures/marker6x6_2.png',
        )

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
    node.get_logger().info("Waiting for joint_states before initial scan...")
    for _ in range(100):
        if node._last_joint_msg is not None:
            break
        time.sleep(0.1)
    else:
        node.get_logger().warn("Timed out waiting for joint_states — proceeding anyway")

    node.get_logger().info("Starting initial scan for markers...")
    node.load_state()

    if runVirtual:
        bad_pos, bad_euler = printer1.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=0, bad_pos=bad_pos, bad_euler=bad_euler)
        bad_pos, bad_euler = printer2.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=1, bad_pos=bad_pos, bad_euler=bad_euler)
        bad_pos, bad_euler = printer3.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=2, bad_pos=bad_pos, bad_euler=bad_euler)

    else:
        viewing_distance = 0.15
        node.marker_offset_config[0] = 'box_offset'
        node.marker_offset_config[1] = 'box_offset'
        node.marker_offset_config[2] = 'printer_offset'

        bad_pos, bad_euler = printer1.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=0, bad_pos=bad_pos, bad_euler=bad_euler)
        bad_pos, bad_euler = printer2.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=1, bad_pos=bad_pos, bad_euler=bad_euler)
        bad_pos, bad_euler = printer3.get_door_marker_pose_in_base()
        node.register_estimated_marker(marker_id=2, bad_pos=bad_pos, bad_euler=bad_euler)

        node.register_printers([
            {"marker_id": 0, "pos": [0.26, -0.3, 0.07], "orient": [0.0, 0.0, np.pi],
             "door_marker_texture": 'materials/textures/marker6x6_0.png'},
            {"marker_id": 1, "pos": [0.48, -0.3, 0.07], "orient": [0.0, 0.0, np.pi],
             "door_marker_texture": 'materials/textures/marker6x6_1.png'},
            {"marker_id": 2, "pos": [0.65, 0.1, 0.07], "orient": [0.0, 0.0, 3/2*np.pi],
             "door_marker_texture": 'materials/textures/marker6x6_2.png'},
        ])

        node.scanMarkerApproach(marker_id=0, viewing_distance=viewing_distance)
        node.scanMarkerApproach(marker_id=1, viewing_distance=viewing_distance)
        node.scanMarkerApproach(marker_id=2, viewing_distance=viewing_distance)

    node.get_logger().info("Initial scan complete.")
    _input_thread(node)


if __name__ == '__main__':
    main()
