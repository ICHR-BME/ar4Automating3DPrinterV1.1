#!/usr/bin/env python3
"""
Hand-guide the physical arm to find markers 1 and 2 instead of typing
coordinates.

Puts the UFACTORY arm in manual (drag-teach) mode, then for each marker you
physically guide the wrist camera until the marker is detected. The detected
pose in base frame is saved to data/manual_marker_estimates.json, which
scanFor2Markers.py loads (use_manual_estimates = 1 there) as the initial
estimates in place of the hardcoded printer coordinates.

Hardware-only: drag-teach mode exists on the xarm driver ('lite6'/'xarm6'),
not in Gazebo and not on the AR4. Launch the arm stack first
(scripts/launchPhysicalXArm6.sh or launchPhysicalXArmLite6.sh).
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R

from xarm_msgs.msg import RobotMsg
from xarm_msgs.srv import Call, SetInt16, SetInt16ById

from ar4_automation.runner_common import (
    make_webcam_node,
    spin_in_background,
    wait_for_joint_states,
)

# ---- config (edit these; no CLI args) ------------------------------------
robot = 'xarm6'          # 'lite6' | 'xarm6' (must match the running launch)
MARKER_IDS = [1, 2]      # captured in this order
CAPTURE_TIMEOUT = 60.0   # s to wait per attempt before offering a retry
FILTER_TIME = 2.0        # s of detection frames fed through the low-pass
                         # filter before the pose is taken (hold still)
FILTER_TAU = 0.5         # s low-pass time constant (first-order IIR, same
                         # approach as web_video_server's display smoothing)

ESTIMATES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "manual_marker_estimates.json"
)

# xarm driver service namespace = hw_ns of the realmove launch
# (xarm6_moveit_realmove defaults to 'xarm', lite6_moveit_realmove to 'ufactory')
HW_NS = {'xarm6': '/xarm', 'lite6': '/ufactory'}

# XARM_MODE constants (xarm_sdk uxbus_cmd_config.h)
MODE_SERVO = 1        # what the ros2_control hardware interface runs in
MODE_TEACH_JOINT = 2  # drag/manual mode
STATE_START = 0


def _call_service(node, client, request, label):
    """Call an xarm service and raise on failure (ret != 0)."""
    if not client.wait_for_service(timeout_sec=5.0):
        raise RuntimeError(f"{label}: service {client.srv_name} unavailable — "
                           f"is the physical launch running (and robot='{robot}' right)?")
    future = client.call_async(request)
    deadline = time.monotonic() + 5.0
    # the background executor completes the future; just wait on it
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.05)
    result = future.result()
    if result is None:
        raise RuntimeError(f"{label}: no response")
    if result.ret != 0:
        raise RuntimeError(f"{label} failed: ret={result.ret} {result.message}")


class ArmModeSwitcher:
    """Switch the xarm controller's mode and VERIFY it took effect.

    A SetInt16 ret=0 only means the command was acked — after an error or a
    controller wedge the mode silently stays put. So this watches the driver's
    robot_states feedback (reported mode/state/err) and raises if the robot
    doesn't actually reach the requested mode."""

    def __init__(self, node, ns):
        self.node = node
        self.set_mode = node.create_client(SetInt16, f"{ns}/set_mode")
        self.set_state = node.create_client(SetInt16, f"{ns}/set_state")
        self.motion_enable = node.create_client(SetInt16ById, f"{ns}/motion_enable")
        self.clean_error = node.create_client(Call, f"{ns}/clean_error")
        self.clean_warn = node.create_client(Call, f"{ns}/clean_warn")
        self.last_report = None
        self._sub = node.create_subscription(
            RobotMsg, f"{ns}/robot_states", self._on_report, 10)

    def _on_report(self, msg):
        self.last_report = msg

    def _report_str(self):
        r = self.last_report
        if r is None:
            return "no robot_states feedback received"
        return f"mode={r.mode}, state={r.state}, err={r.err}"

    def switch_to(self, mode, label, attempts=4):
        # the first attempt right after servo mode often doesn't stick (the
        # ros2_control hardware is still tearing down), so retry quietly and
        # only dump the per-attempt history if every attempt fails
        history = [f"before switch: {self._report_str()}"]
        for attempt in range(1, attempts + 1):
            # same recovery preamble the ros2_control hardware runs on
            # activation: a lingering error (e.g. from a previous session's
            # crash) makes set_mode a silent no-op even though it returns ret=0
            _call_service(self.node, self.clean_error, Call.Request(), "clean_error")
            _call_service(self.node, self.clean_warn, Call.Request(), "clean_warn")
            _call_service(self.node, self.motion_enable,
                          SetInt16ById.Request(id=8, data=1), "motion_enable(all)")
            _call_service(self.node, self.set_mode,
                          SetInt16.Request(data=mode), f"set_mode({mode})")
            _call_service(self.node, self.set_state,
                          SetInt16.Request(data=STATE_START), "set_state(0)")

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                report = self.last_report
                if report is not None and report.mode == mode and report.err == 0:
                    print(f"  Robot confirmed in {label} (mode {mode}, "
                          f"state {report.state}).")
                    return
                time.sleep(0.1)
            history.append(f"attempt {attempt}/{attempts} did not stick: "
                           f"{self._report_str()}")

        raise RuntimeError(
            f"Robot did not enter {label} (wanted mode {mode}) after {attempts} "
            f"attempts:\n  " + "\n  ".join(history) + "\n"
            f"If the UFACTORY Studio web page is open, CLOSE IT — while connected "
            f"it manages the robot mode itself and reverts external mode changes. "
            f"Otherwise check for e-stop/error in Studio, close it, and retry.")


def _lowpass_pose(node, marker_id):
    """Run committed detections of marker_id through a first-order low-pass
    filter for FILTER_TIME and return (pos, euler, n_samples) — the filter
    state at the end of the window. Same IIR approach as web_video_server's
    display smoothing: alpha blend on position, rotvec increment on
    orientation. Each camera frame commits a NEW entry dict into
    found_markers, so identity comparison detects fresh frames."""
    filt_pos = None
    filt_rot = None
    last_entry = None
    last_t = None
    n_samples = 0
    deadline = time.monotonic() + FILTER_TIME
    while time.monotonic() < deadline:
        entry = node.stream.found_markers.get(marker_id)
        if entry is not None and entry is not last_entry and 'positionInBase' in entry:
            last_entry = entry
            now = time.monotonic()
            pos_new = np.asarray(entry['positionInBase'], dtype=float)
            rot_new = R.from_euler("XYZ", entry['eulerInBase'], degrees=False)
            if filt_pos is None:
                filt_pos, filt_rot = pos_new, rot_new
            else:
                dt = now - last_t
                alpha = dt / (FILTER_TAU + dt)
                filt_pos = alpha * pos_new + (1.0 - alpha) * filt_pos
                filt_rot = filt_rot * R.from_rotvec(
                    alpha * (filt_rot.inv() * rot_new).as_rotvec())
            last_t = now
            n_samples += 1
        time.sleep(0.01)
    return filt_pos, filt_rot.as_euler("XYZ", degrees=False), n_samples


def capture_marker(node, marker_id):
    """Open the update window for marker_id and wait for the camera to commit
    a detection while the user hand-guides the arm, then low-pass filter the
    pose over FILTER_TIME. Returns (pos, euler, n_samples) or None on timeout."""
    # drop any previous entry so only a fresh camera commit counts
    node.stream.found_markers.pop(marker_id, None)
    node.allow_marker_updates(marker_id)
    try:
        deadline = time.monotonic() + CAPTURE_TIMEOUT
        last_note = 0.0
        while time.monotonic() < deadline:
            entry = node.stream.found_markers.get(marker_id)
            if entry is not None and 'positionInBase' in entry:
                print(f"  Marker {marker_id} detected — hold the arm still, "
                      f"filtering for {FILTER_TIME:.0f}s...")
                return _lowpass_pose(node, marker_id)
            now = time.monotonic()
            if now - last_note > 5.0:
                remaining = int(deadline - now)
                print(f"  Waiting for marker {marker_id} in the camera view "
                      f"({remaining}s left)...")
                last_note = now
            time.sleep(0.1)
        return None
    finally:
        node.block_marker_updates()


def save_estimates(captured):
    """Write the captured markers to ESTIMATES_PATH (overwrites)."""
    data = {"robot": robot, "markers": captured}
    os.makedirs(os.path.dirname(ESTIMATES_PATH), exist_ok=True)
    with open(ESTIMATES_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(captured)} marker estimate(s) to {ESTIMATES_PATH}")


def main():
    rclpy.init()
    node = make_webcam_node(robot=robot)
    spin_in_background(node)
    wait_for_joint_states(node)

    switcher = ArmModeSwitcher(node, HW_NS[robot])
    # give the robot_states subscription a moment to deliver the first report
    time.sleep(1.0)

    print("\nEntering manual (drag-teach) mode — the arm can then be moved by hand.")
    switcher.switch_to(MODE_TEACH_JOINT, "manual (drag-teach) mode")

    captured = []
    try:
        for marker_id in MARKER_IDS:
            while True:
                input(f"\nGuide the camera to face marker {marker_id}, then press "
                      f"Enter to start looking for it...")
                result = capture_marker(node, marker_id)
                if result is not None:
                    pos, euler, n_samples = result
                    print(f"  Captured marker {marker_id}: pos={np.round(pos, 4)}, "
                          f"euler={np.round(euler, 4)} ({n_samples} frames filtered)")
                    captured.append({
                        "id": int(marker_id),
                        "positionInBase": pos.tolist(),
                        "eulerInBase": euler.tolist(),
                    })
                    # save after every capture — a Ctrl-C later must not lose
                    # markers that were already taught
                    save_estimates(captured)
                    break
                retry = input(f"  Marker {marker_id} was not seen. Retry? [Y/n] ").strip().lower()
                if retry == 'n':
                    print(f"  Skipping marker {marker_id}.")
                    break
    finally:
        # hand control back to MoveIt regardless of how the capture loop ended
        # (Ctrl-C at this prompt must still restore servo mode)
        try:
            input("\nDone guiding. Let go of the arm, then press Enter to re-enable "
                  "servo mode (the arm will hold its current position)...")
        except (KeyboardInterrupt, EOFError):
            print("\n  (interrupted — re-enabling servo mode anyway)")
        switcher.switch_to(MODE_SERVO, "servo mode")

    if not captured:
        print("\nNo markers captured — nothing saved.")
        return
    print(f"\nSaved {len(captured)} marker estimate(s) to {ESTIMATES_PATH}")
    print("Run scanFor2Markers.py (use_manual_estimates = 1) to use them.")


if __name__ == '__main__':
    main()
