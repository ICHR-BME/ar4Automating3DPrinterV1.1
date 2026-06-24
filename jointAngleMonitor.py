#!/usr/bin/env python3
"""Live joint-angle / velocity readout for back-driving the robot by hand.

Subscribes to /joint_states and prints all six joints in degrees, refreshing in
place. Use it to check encoder direction: turn a joint by hand in its + direction
and watch whether the reported position goes UP (correct) or DOWN (inverted), and
whether its velocity sign matches the others for the same physical motion.

Run:  python3 jointAngleMonitor.py
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import time

JOINT_ORDER = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
REFRESH_HZ = 10.0


class JointAngleMonitor(Node):
    def __init__(self):
        super().__init__('joint_angle_monitor')
        self.sub = self.create_subscription(
            JointState, '/joint_states', self.cb, 10)
        self.last_draw = 0.0
        print("waiting for /joint_states ...", flush=True)

    def cb(self, msg):
        now = time.time()
        if now - self.last_draw < 1.0 / REFRESH_HZ:
            return
        self.last_draw = now

        idx = {n: i for i, n in enumerate(msg.name)}
        lines = [f"{'joint':>8} {'pos(deg)':>12} {'vel(deg/s)':>12}",
                 "-" * 36]
        for n in JOINT_ORDER:
            i = idx.get(n)
            if i is None or i >= len(msg.position):
                lines.append(f"{n:>8} {'--':>12} {'--':>12}")
                continue
            pos = np.degrees(msg.position[i])
            vel = np.degrees(msg.velocity[i]) if i < len(msg.velocity) else 0.0
            mark = "   <-- J4 (turn me)" if n == 'joint_4' else ""
            lines.append(f"{n:>8} {pos:12.1f} {vel:12.1f}{mark}")

        # Home cursor, rewrite each line clearing to end-of-line, clear below.
        out = "\033[H" + "".join(line + "\033[K\n" for line in lines) + "\033[J"
        print(out, end="", flush=True)


def main():
    rclpy.init()
    print("\033[2J", end="")  # clear screen once at start
    try:
        rclpy.spin(JointAngleMonitor())
    except KeyboardInterrupt:
        print("\n")


if __name__ == '__main__':
    main()
