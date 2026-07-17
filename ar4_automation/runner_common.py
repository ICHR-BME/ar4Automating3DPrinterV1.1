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

import math
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

# Gazebo configuration: camera images come from the simulated RGBD camera
# bridged by annin_ar4_gazebo (topics /rgbd_camera/image + camera_info, so no
# webcam calibration file or distance-scale correction applies).
SIM_NODE_KWARGS = dict(
    calibration_mode=False,
    stream_source="ros",
)

# Standard simulated printer layouts (poses match the Gazebo table setup used
# by the scanForNMarkers scripts). Marker IDs correspond to the door textures.
SIM_PRINTER_SPECS_3 = [
    {"marker_id": 0, "pos": [0.22, -0.2, 0.21], "orient": [0.0, 0.0, math.pi],
     "door_marker_texture": 'materials/textures/marker6x6_0.png'},
    # y=-0.3 (matching the hardware layout's lateral offset) keeps the 0.38 m
    # scrape standoff inside the arm's wrist envelope; at y=-0.2 the standoff
    # pose has no IK solution and scrapePlate aborts at the scrape waypoints.
    {"marker_id": 1, "pos": [0.44, -0.3, 0.21], "orient": [0.0, 0.0, math.pi],
     "door_marker_texture": 'materials/textures/marker6x6_1.png'},
    {"marker_id": 2, "pos": [0.60, 0.1, 0.21], "orient": [0.0, 0.0, 3/2*math.pi],
     "door_marker_texture": 'materials/textures/marker6x6_2.png'},
]
SIM_PRINTER_SPECS_2 = SIM_PRINTER_SPECS_3[1:]


def make_webcam_node(**overrides):
    """Build a printerAutomation node with the standard webcam config."""
    kwargs = dict(WEBCAM_NODE_KWARGS)
    kwargs.update(overrides)
    node = printerAutomation(**kwargs)
    node.stream.distance_scale = WEBCAM_DISTANCE_SCALE
    return node


def make_sim_node(**overrides):
    """Build a printerAutomation node fed by the Gazebo camera."""
    kwargs = dict(SIM_NODE_KWARGS)
    kwargs.update(overrides)
    node = printerAutomation(**kwargs)
    # Gripper action is unstable in sim (see printerAutomation.gripper_disabled)
    node.gripper_disabled = True
    return node


def start_node(sim=False, **overrides):
    """(make_webcam_node or make_sim_node) + spin_in_background + wait_for_joint_states."""
    node = make_sim_node(**overrides) if sim else make_webcam_node(**overrides)
    spin_in_background(node)
    wait_for_joint_states(node)
    return node


def spawn_sim_printers(node, specs):
    """
    Spawn Simulated3DPrinter models in Gazebo (one per spec) and register each
    door marker's geometric pose as the node's estimated marker, so scans know
    where to look. Returns the list of Simulated3DPrinter objects.
    """
    printers = []
    for p in specs:
        printer = Simulated3DPrinter(
            node=node,
            pos=p["pos"],
            orient=p["orient"],
            door_marker_texture=p["door_marker_texture"],
        )
        printer.spawn_fast()
        bad_pos, bad_euler = printer.get_door_marker_pose_in_base()
        node.register_estimated_marker(
            marker_id=p["marker_id"], bad_pos=bad_pos, bad_euler=bad_euler
        )
        printers.append(printer)
    return printers


def spin_in_background(node):
    """Start a MultiThreadedExecutor for the node (and its stream) in a daemon thread."""
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    if hasattr(node.stream, "_ros_node"):
        executor.add_node(node.stream._ros_node)
    else:
        threading.Thread(target=node.stream.run, daemon=True).start()

    def _resilient_spin():
        # spin() (not a spin_once loop): a spin_once loop serves one callback per
        # iteration, so the always-ready 30 Hz sim-camera callback starves TF,
        # joint_states, and MoveIt action results — verification then compares
        # against a frozen pose and every sim move "times out" despite executing.
        while rclpy.ok():
            try:
                executor.spin()
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
    print("  2) Walk pickup waypoints (scan/gripper entries included)")
    print("  3) Pickup plate (walk pickup list, record grasp for replay)")
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
            node.get_logger().info(
                f"User requested transferPlate({source_id}, {dest_id}, {rescan_id})"
            )
            node.transferPlate(source_id=source_id, dest_id=dest_id, rescan_id=rescan_id)

        elif choice == "9":
            ids = _parse_floats("  Source, scrape marker IDs (e.g. 1 2): ", 2)
            if ids is None:
                continue
            source_id, scrape_id = int(ids[0]), int(ids[1])
            # All motion (scans included) comes from the offset-config
            # waypoint lists.
            node.get_logger().info(
                f"User requested scrapePlate({source_id}, {scrape_id})"
            )
            node.scrapePlate(source_id=source_id, scrape_id=scrape_id)

        else:
            print("  Unknown option. Try again.")
