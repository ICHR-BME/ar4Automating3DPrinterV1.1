#!/usr/bin/env python3
"""Shared setup/menu boilerplate for the run*/scanFor* entry scripts."""

import math
import os
import socket
import subprocess
import time
import threading

import rclpy

from .printer_automation import printerAutomation, MarkerNotVisibleError
from .simulated3DPrinter import Simulated3DPrinter
from .marker_sources import (
    MANUAL, NONE, SCAN, load_marker_poses, source_path,
)

# Marker edge length (m) per ArUco dictionary, indexed like the detector's
# dict_names: [DICT_4X4_50, DICT_6X6_50]. First entry: 0.024 m measured with
# calipers 2026-07-24 (black square side). NB calibrate_camera_offset's
# depth-scale fit implied an effective 0.0271 — the leftover ~13% is elsewhere
# (likely camera intrinsics), not the marker size.
# SIM USES THE SAME VALUES on purpose: the spawned plate takes its size from the
# mount in models/printers/*.json (0.025), so a sim scan and a hardware scan of
# the same marker must resolve to the same distance. Estimated range scales
# linearly with this number — the old sim-only 0.05 default put every marker
# twice as far away as it really was.
MARKER_SIZES = [0.024, 0.025]

# standard hardware config for all runner scripts
WEBCAM_NODE_KWARGS = dict(
    calibration_mode=False,
    stream_source="webcam",
    feed_rotation_deg=90.0,
    marker_sizes=MARKER_SIZES,
)

# Gazebo: images come from the bridged RGBD camera on /rgbd_camera/*, so no
# webcam calibration file or distance-scale correction here.
SIM_NODE_KWARGS = dict(
    calibration_mode=False,
    stream_source="ros",
    marker_sizes=MARKER_SIZES,
)

# sim printer layouts per robot. pos/orient are the PRINTER BODY pose (its box
# model's center) in the robot's good frame; marker_id is the ArUco ID its
# 'door' mount wears, and printer_model names the box model in models/printers/.
# Where that marker then appears is derived from the mount recorded in the
# model's JSON — see Simulated3DPrinter.marker_pose_in_base.
DEFAULT_SIM_PRINTER_MODEL = 'a1_mini'
# the mount (a name in the model JSON's "markers") a spec's marker_id refers to
MOUNT = Simulated3DPrinter.DEFAULT_MOUNT

SIM_PRINTER_SPECS = {
    'ar4': [
        {"marker_id": 0, "pos": [0.22, -0.2, 0.21], "orient": [0.0, 0.0, math.pi]},
        # y must stay -0.3: at -0.2 the 0.38m scrape standoff has no IK solution
        # and scrapePlate aborts at the scrape waypoints
        {"marker_id": 1, "pos": [0.44, -0.3, 0.21], "orient": [0.0, 0.0, math.pi]},
        {"marker_id": 2, "pos": [0.60, 0.1, 0.21], "orient": [0.0, 0.0, 3/2*math.pi]},
    ],
    # tuned in sim for the lite6's 0.44 m reach: the far (1.75x) viewing pose
    # of each marker must stay reachable even with the 0.03 m estimate
    # randomization. Reach-probed grid: door-facing-+y poses only plan
    # reliably for marker x <= ~0.30 with y around -0.24 (at 0.263 m standoff).
    # Davis layout (inactive — swap with the Monterrey block below to use):
    # 'lite6': [
    #     {"marker_id": 0, "pos": [0.14, -0.16, 0.15], "orient": [0.0, 0.0, math.pi]},
    #     {"marker_id": 1, "pos": [0.3, -0.3, 0.18], "orient": [0.0, 0.0, math.pi]},
    #     {"marker_id": 2, "pos": [0.58, 0.08, 0.20], "orient": [0.0, 0.0, 3/2*math.pi]},
    # ],
    'lite6': [ #Monterrey
            {"marker_id": 0, "pos": [0.14, -0.16, 0.15], "orient": [0.0, 0.0, math.pi]},
            {"marker_id": 2, "pos": [0.3, -0.45, 0.24], "orient": [0.0, 0.0, 0*math.pi]},
            {"marker_id": 1, "pos": [0.40, 0.0, -0.1], "orient": [-1/2*math.pi, 0.0, 2/2*math.pi]},
        ],
        # {"marker_id": 2, "pos": [0.45, -0.2, 0.12], "orient": [0.0, 0.0, -1/2*math.pi]},
    # xArm 6: same door-facing conventions as the lite6, pushed out for the
    # longer (~0.7 m) reach. Starting point — re-probe reachability of each
    # marker's far viewing pose on the first sim runs and nudge as needed.
    # Full-size A1 printers (not the mini): the xArm 6 is the only arm with
    # the reach for them, so the model is pinned per spec here rather than
    # via DEFAULT_SIM_PRINTER_MODEL (ar4/lite6 layouts stay a1_mini).
    # pos z = 0.30 puts the A1's base (height 0.605, so half-height ~0.302)
    # on the floor; its door marker then sits ~0.61 m up — re-probe the far
    # viewing poses for reachability on the first runs.
    'xarm6': [
            {"marker_id": 0, "pos": [0.20, -0.22, 0.30], "orient": [0.0, 0.0, math.pi],
             "printer_model": "a1"},
            {"marker_id": 1, "pos": [0.30, 0.35, 0.30], "orient": [0.0, 0.0, 0*math.pi],
             "printer_model": "a1"},
            {"marker_id": 2, "pos": [0.70, 0.10, 0.30], "orient": [0.0, 0.0, 3/2*math.pi],
             "printer_model": "a1"},
        ],
}


def sim_printer_specs(robot='ar4', count=3):
    """The robot's sim printer layout; count=2 drops marker 0 (ids 1+2)."""
    return SIM_PRINTER_SPECS[robot][-count:]


# legacy names (AR4 layout)
SIM_PRINTER_SPECS_3 = SIM_PRINTER_SPECS['ar4']
SIM_PRINTER_SPECS_2 = SIM_PRINTER_SPECS_3[1:]


# The gz world name the sim launch scripts load (empty_gz.world).
GZ_WORLD = 'default'


def ensure_gz_ip():
    """Make sure gz-transport can talk to itself on this host, WITHOUT
    hardcoding an address. gz binds its sockets to the first non-loopback
    interface; on some networks host-local traffic on that interface is
    dropped (seen with wifi that hands out CGNAT 100.64/10 addresses while
    Tailscale's firewall claims the same range) and then every gz service
    call — model spawns included — hangs forever. Rather than looking for
    that one culprit, empirically test the interface gz would pick: bind a
    listening socket to it and try to connect. Only if that self-connection
    fails is GZ_IP pinned to loopback; on machines where the default works,
    nothing changes. An existing GZ_IP (e.g. from a launch script) is
    respected either way."""
    if os.environ.get('GZ_IP'):
        return os.environ['GZ_IP']
    # The IP gz-transport would choose: source address of the default route.
    # Connecting a UDP socket sends no packets; it just resolves routing.
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(('8.8.8.8', 53))
        candidate = probe.getsockname()[0]
        probe.close()
    except OSError:
        return None  # no route at all; gz will use loopback on its own
    try:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind((candidate, 0))
        srv.listen(1)
        cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cli.settimeout(1.0)
        cli.connect((candidate, srv.getsockname()[1]))
        cli.close()
        srv.close()
        return None  # interface is healthy; leave gz on its default
    except OSError:
        os.environ['GZ_IP'] = '127.0.0.1'
        return '127.0.0.1'


# Invisible static box whose TOP face sits at z=0, standing in for the floor.
# The world's ground_plane is visual-only (see empty_gz.world: scanned markers
# can sit below z=0, and a solid floor wedges the arm on scrape waypoints), so
# floor physics is opt-in: collisions=True runs spawn this.
GZ_FLOOR_SDF = """<?xml version="1.0"?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <pose>0 0 -0.05 0 0 0</pose>
      <collision name="collision">
        <geometry><box><size>100 100 0.1</size></box></geometry>
      </collision>
    </link>
  </model>
</sdf>"""


def add_gz_floor_collision(node, world=GZ_WORLD, name='floor_collision'):
    """Spawn the collision-only floor box (top face at z=0). The visual floor
    in empty_gz.world carries no collision of its own, so this is what makes
    the ground solid in collisions=True sim runs. A second call in the same gz
    session finds the model already there — that counts as success. Returns
    True when the floor is solid."""
    sdf = GZ_FLOOR_SDF.format(name=name)
    req = f'sdf: {_gz_quote(sdf)}, name: "{name}"'
    try:
        out = subprocess.run(
            ['gz', 'service', '-s', f'/world/{world}/create',
             '--reqtype', 'gz.msgs.EntityFactory', '--reptype', 'gz.msgs.Boolean',
             '--timeout', '4000', '--req', req],
            capture_output=True, text=True, timeout=8)
        ok = 'true' in out.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        ok = False
    log = node.get_logger()
    if ok:
        log.info(f'collisions=True: spawned gz floor collision box "{name}".')
    else:
        # create returns false for duplicate names; treat "already spawned
        # this session" as solid rather than warning every rerun.
        log.info(f'collisions=True: gz floor box "{name}" not created '
                 '(already present from an earlier run, or gz unreachable).')
    return ok


def _gz_quote(s):
    """Quote a string for a protobuf text-format field."""
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'


def make_webcam_node(robot='ar4', **overrides):
    """Build a printerAutomation node with the standard webcam config."""
    kwargs = dict(WEBCAM_NODE_KWARGS)
    kwargs.update(overrides)
    node = printerAutomation(robot=robot, **kwargs)
    return node


def make_sim_node(robot='ar4', **overrides):
    """Build a printerAutomation node fed by the Gazebo camera."""
    kwargs = dict(SIM_NODE_KWARGS)
    kwargs.update(overrides)
    node = printerAutomation(robot=robot, **kwargs)
    # The AR4's GripperCommand action and the Lite 6's driver services don't work
    # in sim, so gripper commands are skipped there. The xArm 6's gripper is a
    # plain JointTrajectoryController that DOES work in sim (verified), and scans
    # need it closed to keep the jaws out of the camera's view.
    node.gripper_disabled = (robot != 'xarm6')
    return node


def start_node(sim=False, robot='ar4', collisions=True, **overrides):
    """(make_webcam_node or make_sim_node) + spin_in_background + wait_for_joint_states.

    robot: 'ar4', 'lite6', or 'xarm6' (see robot_config.py). Sim launch scripts:
    launchVirtualRobot.sh for the AR4, launchVirtualXArmLite6.sh for the Lite 6,
    launchVirtualXArm6.sh for the xArm 6.

    collisions: False turns collisions off in ALL the places they exist.
    (1) MoveIt: plan in an EMPTY world — no ground plane, no printer boxes
    (spawn_sim_printers / spawn_printers_from_markers add nothing either), AND
    anything an earlier run left in move_group's world is purged. That purge
    matters: move_group belongs to the launch file, not to this script, so it
    keeps its scene between runs — without it, "collisions off" would still plan
    against the boxes the last collisions-on run published.
    (2) Gazebo: printers spawn visual-only (Simulated3DPrinter reads this off
    the node), so the arm passes through them instead of stalling against solid
    geometry and aborting the trajectory. A MoveIt-only switch leaves the
    physical obstruction in place, which fails in a way that looks unrelated.
    (3) The gz floor: empty_gz.world's ground plane is VISUAL-ONLY, so with
    collisions off, waypoints that dip the gripper below z=0 pass through
    instead of wedging the arm against a solid floor (CONTROL_FAILED aborts).
    collisions=True spawns a collision-only floor box (add_gz_floor_collision)
    to give the ground physics back.
    Set it here rather than after this returns: the ground plane goes in below,
    so flipping node.collision_scene_enabled afterwards comes too late for it.
    Off means nothing stops a path sweeping the EEF through the floor or a
    printer, so it is a diagnostic — it separates "the plan is in collision"
    from "the goal is unreachable" — not a way to run an unattended job.
    """
    if sim:
        # Before anything gz-flavored starts: the spawn bridge subprocess and
        # every `gz service` call inherit this environment.
        ensure_gz_ip()
    node = make_sim_node(robot=robot, **overrides) if sim else make_webcam_node(robot=robot, **overrides)
    node.collision_scene_enabled = bool(collisions)
    spin_in_background(node)
    wait_for_joint_states(node)
    if not collisions:
        node.get_logger().warn(
            "collisions=False: planning in an empty world (no ground plane, no "
            "printer boxes). Paths may pass through the floor or a printer.")
        node.purge_collision_scene()
    elif sim:
        # The world's floor is visual-only; collisions=True makes it solid.
        add_gz_floor_collision(node)
    # move_group's world starts EMPTY — without this the planner happily sweeps
    # the EEF through the floor (printer boxes go in as each printer is spawned)
    node.add_ground_plane()
    return node


def make_sim_printer(node, spec, **overrides):
    """One Simulated3DPrinter from a layout spec (marker_id/pos/orient[/printer_model])."""
    kwargs = dict(
        node=node,
        pos=spec["pos"],
        orient=spec["orient"],
        printer_model=spec.get("printer_model", DEFAULT_SIM_PRINTER_MODEL),
        marker_ids={MOUNT: spec["marker_id"]},
    )
    kwargs.update(overrides)
    return Simulated3DPrinter(**kwargs)


def spawn_sim_printers(node, specs):
    """Spawn a sim printer per spec, at the spec's body pose, and register where
    its door marker therefore is so scans know where to look. Use this when the
    SPAWNED printers are the ground truth (pure-sim rehearsal); use
    spawn_printers_from_markers when a marker file is."""
    printers = []
    for p in specs:
        printer = make_sim_printer(node, p)
        printer.spawn()
        printer.register_marker_estimates(node)
        # same boxes Gazebo just spawned, now in the planning scene
        node.add_printer_collision_boxes(printer)
        # this marker's waypoints come from this printer model's JSON
        node.marker_offset_config[p["marker_id"]] = printer.printer_model.name
        printers.append(printer)
    return printers


def spawn_printers_from_markers(node, specs, source=SCAN, require_detected=True,
                                mount=None, fallback=True, register=False):
    """Spawn each spec's printer where a MARKER FILE says its marker is.

    The inverse of spawn_sim_printers: instead of deriving marker poses from
    printer poses, this back-solves each printer body from an observed marker
    plus the mount offset in its model JSON. Use it to see the real layout in
    Gazebo, and to put it in the planning scene (each spawned printer's boxes are
    added there too, so the planner avoids what you see).

    source: marker_sources.SCAN (data/printer_state.json, written by
        scanFor2Markers.py / scanFor3Markers.py), marker_sources.MANUAL
        (data/manual_marker_estimates.json, written by teachMarkersByHand.py),
        or a path to a file in either format.
    require_detected: with SCAN, ignore markers flagged 'estimated' — poses a
        scan never confirmed. Pass False to place printers from seeds too.
    fallback: when a spec's marker isn't in the file, spawn it at the spec's own
        pos/orient anyway (False = skip it).
    register: also register each spawned printer's mount pose as a marker
        estimate on the node.

    Returns the list of spawned printers.
    """
    mount = mount or MOUNT
    log = node.get_logger().info if hasattr(node, 'get_logger') else None
    poses = load_marker_poses(source, require_detected=require_detected, log=log)

    printers = []
    for p in specs:
        marker_id = p["marker_id"]
        model = p.get("printer_model", DEFAULT_SIM_PRINTER_MODEL)
        if marker_id in poses:
            printer = Simulated3DPrinter.from_marker(
                *poses[marker_id], printer_model=model, mount=mount,
                node=node, marker_ids={mount: marker_id})
            origin = f"marker {marker_id} from {os.path.basename(source_path(source))}"
        elif fallback:
            printer = make_sim_printer(node, p, printer_model=model)
            origin = "spec pose (marker not in the file)"
        else:
            if log:
                log(f"marker {marker_id}: not in the marker file — printer skipped")
            continue
        printer.spawn()
        if log:
            base_z = float(printer.pos[2]) - printer.height / 2.0
            log(f"printer for marker {marker_id} placed from {origin}; "
                f"base sits at z={base_z:+.3f} m (0 = on the floor)")
        if register:
            printer.register_marker_estimates(node)
        node.add_printer_collision_boxes(printer)
        # this marker's waypoints come from this printer model's JSON
        node.marker_offset_config[marker_id] = printer.printer_model.name
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
        # must be spin(), not a spin_once loop: the always-ready 30 Hz sim camera
        # callback would starve TF/joint_states/MoveIt results and every sim move
        # "times out" despite executing
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


def start_webcam_node(robot='ar4', collisions=True, **overrides):
    """make_webcam_node + spin_in_background + wait_for_joint_states.

    collisions: see start_node — False plans in an empty world."""
    node = make_webcam_node(robot=robot, **overrides)
    node.collision_scene_enabled = bool(collisions)
    spin_in_background(node)
    wait_for_joint_states(node)
    if not collisions:
        node.get_logger().warn(
            "collisions=False: planning in an empty world (no ground plane, no "
            "printer boxes). Paths may pass through the floor or a printer.")
        node.purge_collision_scene()
    node.add_ground_plane()
    return node


def restore_saved_printers(node):
    """Rebuild printers from load_state() configs (not spawned); their door-mount
    poses back-fill markers with no real saved pose."""
    for p in getattr(node, "_saved_printer_configs", []):
        printer = make_sim_printer(node, p)
        bad_pos, bad_euler = printer.marker_pose_in_base(MOUNT)
        existing = node._find_marker_entry(p["marker_id"])
        if existing is None or existing.get("estimated"):
            node.register_estimated_marker(
                marker_id=p["marker_id"], bad_pos=bad_pos, bad_euler=bad_euler
            )
        # this marker's waypoints come from this printer model's JSON
        node.marker_offset_config[p["marker_id"]] = printer.printer_model.name


# hand-taught estimates written by teachMarkersByHand.py
MANUAL_ESTIMATES_PATH = source_path(MANUAL)


def register_manual_estimates(node, marker_ids=None):
    """Register the hand-taught marker estimates from teachMarkersByHand.py
    as initial scan seeds (call after load_state so stale saved poses can't
    shadow them). marker_ids, when given, limits which IDs are taken from the
    file. Returns the list of marker IDs registered — empty if the file is
    missing or unreadable (a warning is logged; the caller falls back to its
    own seeds)."""
    poses = load_marker_poses(MANUAL, log=node.get_logger().info)
    if not poses:
        node.get_logger().warn(
            f"No hand-taught estimates in {MANUAL_ESTIMATES_PATH}. "
            f"Run teachMarkersByHand.py to create them."
        )
        return []

    registered = []
    for marker_id, (pos, euler) in poses.items():
        if marker_ids is not None and marker_id not in marker_ids:
            continue
        node.register_estimated_marker(
            marker_id=marker_id, bad_pos=pos, bad_euler=euler)
        registered.append(marker_id)
    return registered


def require_scanned_markers(node, marker_ids):
    """True iff every marker has a real scanned (non-estimated) saved pose.

    The run* scripts that move on markers without their own initial scan must
    work from measurements made by an official scan procedure
    (scanFor2Markers.py / scanFor3Markers.py) — hand-taught or geometric
    estimates are not accepted there. Logs an error naming the offenders."""
    missing = [mid for mid in marker_ids
               if (entry := node._find_marker_entry(mid)) is None
               or entry.get('estimated', False)]
    if missing:
        node.get_logger().error(
            f"Marker(s) {missing} have no scanned pose in the save file — "
            f"run scanFor2Markers.py / scanFor3Markers.py first. Estimates "
            f"(hand-taught or geometric) are not accepted by this script."
        )
        return False
    return True


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
    """Prompt for space-separated floats, None on bad input."""
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
    """Stdin command loop; node must already be spinning in the background. Blocks until EOF."""
    time.sleep(5.0)
    print("\n[INFO] System ready. Type a command number.")
    node.record_startup_time()

    while rclpy.ok():
        _print_menu()
        try:
            choice = input(">> ").strip()
        except EOFError:
            break

        # ROS log output can interleave with terminal input and leave stray
        # leading chars before the digit the user typed
        _valid_choices = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
        if choice not in _valid_choices and len(choice) >= 2 and choice[1:] in _valid_choices:
            choice = choice[1:]

        try:
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
                # all motion (scans included) comes from the offset-config waypoint lists
                node.get_logger().info(
                    f"User requested scrapePlate({source_id}, {scrape_id})"
                )
                node.scrapePlate(source_id=source_id, scrape_id=scrape_id)

            else:
                print("  Unknown option. Try again.")
        except MarkerNotVisibleError as ex:
            node.get_logger().error(f"Command aborted — marker not visible: {ex}")
