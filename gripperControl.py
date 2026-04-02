#!/usr/bin/env python3

import time
from threading import Thread

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from pymoveit2 import GripperInterface


def main():
    rclpy.init()

    node = Node("gripper_control")
    callback_group = ReentrantCallbackGroup()

    gripper = GripperInterface(
        node=node,
        gripper_joint_names=["gripper_jaw1_joint"],
        open_gripper_joint_positions=[0.012],
        closed_gripper_joint_positions=[0.025],
        gripper_group_name="ar_gripper",
        callback_group=callback_group,
        gripper_command_action_name="gripper_controller/gripper_cmd",
    )

    # Spin in background
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # Wait for initialization
    time.sleep(1.0)

    for i in range(10):
        

        node.get_logger().info(f"Opening gripper (iteration {i + 1})")
        gripper.open()
        gripper.wait_until_executed()
        time.sleep(0.3)

        node.get_logger().info(f"Closing gripper (iteration {i + 1})")
        gripper.close()
        gripper.wait_until_executed()
        time.sleep(0.3)

    node.get_logger().info("Done!")
    rclpy.shutdown()
    executor_thread.join()


if __name__ == "__main__":
    main()
