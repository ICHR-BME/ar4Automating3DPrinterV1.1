#!/usr/bin/env python3
"""
Cycle a robot's gripper open/close for testing.

Set MODE below:
  'ar4'   - MoveIt GripperCommand action (pymoveit2 GripperInterface)
  'lite6' - stock UFACTORY Lite 6 gripper via the xarm driver's empty-request
            open/close services (/ufactory/open_lite6_gripper,
            /ufactory/close_lite6_gripper). Needs the physical Lite 6 driver
            running (launchPhysicalXArmLite6.sh) AND those services enabled in
            xarm_user_params.yaml (open/close_lite6_gripper: true).
"""

import time
from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

# ---- Configuration ----
MODE       = 'lite6'   # 'ar4' | 'lite6'
ITERATIONS = 10
PAUSE      = 0.3       # seconds between open and close


def _make_ar4_gripper(node, callback_group):
    """Return (open_fn, close_fn) driving the AR4 gripper over its action."""
    from pymoveit2 import GripperInterface

    # the first values were .011 for open and .024 for closed
    gripper = GripperInterface(
        node=node,
        gripper_joint_names=["gripper_jaw1_joint"],
        open_gripper_joint_positions=[0.000],
        closed_gripper_joint_positions=[0.0145],  # use .026 for full closure
        gripper_group_name="ar_gripper",
        callback_group=callback_group,
        gripper_command_action_name="gripper_controller/gripper_cmd",
    )

    def _open():
        gripper.open()
        gripper.wait_until_executed()

    def _close():
        gripper.close()
        gripper.wait_until_executed()

    return _open, _close


def _make_lite6_gripper(node, callback_group):
    """Return (open_fn, close_fn) calling the xarm Lite 6 gripper services."""
    from xarm_msgs.srv import Call

    open_cli = node.create_client(
        Call, '/ufactory/open_lite6_gripper', callback_group=callback_group)
    close_cli = node.create_client(
        Call, '/ufactory/close_lite6_gripper', callback_group=callback_group)

    def _call(client, label, timeout=5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            node.get_logger().error(
                f"{label}: service {client.srv_name} unavailable "
                "(is the xarm driver running?).")
            return
        future = client.call_async(Call.Request())
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.02)
        if not future.done():
            node.get_logger().error(f"{label}: service call timed out.")
            return
        resp = future.result()
        if resp is not None and getattr(resp, 'ret', 0) != 0:
            node.get_logger().warn(
                f"{label}: driver returned ret={resp.ret} "
                f"msg='{getattr(resp, 'message', '')}'.")

    return (lambda: _call(open_cli, "open_lite6_gripper"),
            lambda: _call(close_cli, "close_lite6_gripper"))


def main():
    rclpy.init()

    node = Node("gripper_control")
    callback_group = ReentrantCallbackGroup()

    if MODE == 'lite6':
        open_gripper, close_gripper = _make_lite6_gripper(node, callback_group)
    elif MODE == 'ar4':
        open_gripper, close_gripper = _make_ar4_gripper(node, callback_group)
    else:
        raise ValueError(f"Unknown MODE '{MODE}'; use 'ar4' or 'lite6'.")

    # Spin in background
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # Wait for initialization
    time.sleep(1.0)

    for i in range(ITERATIONS):
        node.get_logger().info(f"Opening gripper (iteration {i + 1})")
        open_gripper()
        time.sleep(PAUSE)

        node.get_logger().info(f"Closing gripper (iteration {i + 1})")
        close_gripper()
        time.sleep(PAUSE)

    node.get_logger().info("Done!")
    rclpy.shutdown()
    executor_thread.join()


if __name__ == "__main__":
    main()
