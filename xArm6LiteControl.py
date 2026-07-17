#!/usr/bin/env python3
"""
Control the UFACTORY Lite 6 simulated in Gazebo, through MoveIt.

Start the sim first: ./scripts/launchVirtualXArmLite6.sh
(wraps: ros2 launch xarm_moveit_config lite6_moveit_gazebo.launch.py)
Units are meters and radians.
"""

import math
import threading
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from pymoveit2 import MoveIt2

# ---- Configuration ----
VELOCITY_SCALING = 0.2      # fraction of joint velocity limits
ACCELERATION_SCALING = 0.2  # fraction of joint acceleration limits


def main():
    rclpy.init()
    node = Node("xarm_lite6_control")
    moveit2 = MoveIt2(
        node=node,
        joint_names=[f"joint{i}" for i in range(1, 7)],
        base_link_name="link_base",
        end_effector_name="link_eef",
        group_name="lite6",
        callback_group=ReentrantCallbackGroup(),
    )
    moveit2.max_velocity = VELOCITY_SCALING
    moveit2.max_acceleration = ACCELERATION_SCALING

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    time.sleep(2.0)  # let action clients discover the MoveIt servers

    node.get_logger().info("moving to home (all joints zero)")
    moveit2.move_to_configuration([0.0] * 6)
    moveit2.wait_until_executed()

    # pose goal: meters, quaternion xyzw ([1,0,0,0] = tool pointing down)
    node.get_logger().info("moving to pose goal")
    moveit2.move_to_pose(position=[0.25, 0.0, 0.15], quat_xyzw=[1.0, 0.0, 0.0, 0.0])
    moveit2.wait_until_executed()

    node.get_logger().info("joint goal")
    moveit2.move_to_configuration([0.0, math.radians(-30), math.radians(30),
                                   0.0, math.radians(45), 0.0])
    moveit2.wait_until_executed()

    node.get_logger().info("done")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
