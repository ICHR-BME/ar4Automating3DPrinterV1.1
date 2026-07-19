#!/usr/bin/env python3
"""Publish the robot's reachable workspace as a latched PointCloud2 for RViz.

Samples the arm's joint space, runs vectorized forward kinematics (pure numpy,
built from the live URDF on /robot_description), and publishes the cloud of
reachable end-effector positions on /robot_workspace in the base frame.

Run the sim (or the real robot's robot_state_publisher) first, then:
    /bin/python3 tools/workspace_cloud.py            # lite6 (default)
    /bin/python3 tools/workspace_cloud.py --robot ar4 --samples 40000

In RViz: Add -> By topic -> /robot_workspace -> PointCloud2, and set the
Fixed Frame to the robot's base (link_base for lite6, base_link for ar4).
Leave this node running; the topic is latched so it survives RViz restarts.
"""

import argparse
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from urdf_parser_py.urdf import URDF

# base/tip frames per robot (match robot_config.py)
FRAMES = {
    'lite6': ('link_base', 'link_eef'),
    'ar4':   ('base_link', 'link_6'),
}


def rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])


def axis_angle_batch(axis, angles):
    """(N,) angles about a fixed unit axis -> (N,3,3) rotation matrices."""
    ax = axis / np.linalg.norm(axis)
    x, y, z = ax
    c = np.cos(angles)
    s = np.sin(angles)
    C = 1.0 - c
    N = angles.shape[0]
    R = np.empty((N, 3, 3))
    R[:, 0, 0] = c + x*x*C
    R[:, 0, 1] = x*y*C - z*s
    R[:, 0, 2] = x*z*C + y*s
    R[:, 1, 0] = y*x*C + z*s
    R[:, 1, 1] = c + y*y*C
    R[:, 1, 2] = y*z*C - x*s
    R[:, 2, 0] = z*x*C - y*s
    R[:, 2, 1] = z*y*C + x*s
    R[:, 2, 2] = c + z*z*C
    return R


def build_chain(robot, base, tip):
    """Return the list of (origin_T 4x4, axis or None, lower, upper) for each
    joint from base to tip. axis=None marks a fixed joint (no DOF)."""
    chain_names = robot.get_chain(base, tip, joints=True, links=False)
    segments = []
    for jname in chain_names:
        jnt = robot.joint_map[jname]
        xyz = np.array(jnt.origin.xyz if jnt.origin and jnt.origin.xyz else [0, 0, 0])
        rpy = np.array(jnt.origin.rpy if jnt.origin and jnt.origin.rpy else [0, 0, 0])
        T = np.eye(4)
        T[:3, :3] = rpy_to_matrix(rpy)
        T[:3, 3] = xyz
        if jnt.type in ('revolute', 'continuous'):
            axis = np.array(jnt.axis if jnt.axis else [0, 0, 1], dtype=float)
            if jnt.type == 'continuous' or jnt.limit is None:
                lo, hi = -np.pi, np.pi
            else:
                lo, hi = jnt.limit.lower, jnt.limit.upper
            segments.append((T, axis, lo, hi))
        else:  # fixed / other -> no DOF, just the offset transform
            segments.append((T, None, 0.0, 0.0))
    return segments


def forward_kinematics(segments, q_samples):
    """q_samples: (N, dof). Returns (N,3) eef positions in the base frame."""
    N = q_samples.shape[0]
    T = np.broadcast_to(np.eye(4), (N, 4, 4)).copy()
    dof_i = 0
    for origin_T, axis, _, _ in segments:
        T = T @ origin_T  # apply the fixed joint origin offset
        if axis is not None:
            R = axis_angle_batch(axis, q_samples[:, dof_i])
            Tj = np.broadcast_to(np.eye(4), (N, 4, 4)).copy()
            Tj[:, :3, :3] = R
            T = T @ Tj
            dof_i += 1
    return T[:, :3, 3]


class WorkspaceCloud(Node):
    def __init__(self, robot, n_samples):
        super().__init__('workspace_cloud')
        base, tip = FRAMES[robot]
        self.base_frame = base

        urdf = self._get_urdf()
        robot_model = URDF.from_xml_string(urdf)
        segments = build_chain(robot_model, base, tip)
        dof = sum(1 for _, a, _, _ in segments if a is not None)
        self.get_logger().info(f"{robot}: chain {base} -> {tip}, {dof} DOF, sampling {n_samples} configs")

        lows = np.array([lo for _, a, lo, _ in segments if a is not None])
        highs = np.array([hi for _, a, _, hi in segments if a is not None])
        rng = np.random.default_rng(0)  # fixed seed: deterministic cloud
        q = rng.uniform(lows, highs, size=(n_samples, dof))
        pts = forward_kinematics(segments, q)

        self.get_logger().info(
            f"reach: x[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
            f"y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
            f"z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}] m")

        self._cloud = self._make_cloud(pts)
        # TRANSIENT_LOCAL latches for transient-local subscribers; the 1 Hz
        # timer additionally feeds RViz's default VOLATILE subscriber, which
        # otherwise never sees a one-shot message published before it connected
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(PointCloud2, '/robot_workspace', qos)
        self.create_timer(1.0, self._republish)
        self._republish()
        self.get_logger().info("Publishing /robot_workspace at 1 Hz. Add it as PointCloud2 in RViz.")

    def _republish(self):
        self._cloud.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self._cloud)

    def _get_urdf(self):
        """Grab the URDF from the latched /robot_description topic."""
        holder = {}
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST)
        sub = self.create_subscription(
            String, '/robot_description', lambda m: holder.setdefault('urdf', m.data), qos)
        import time
        t0 = time.time()
        while 'urdf' not in holder and time.time() - t0 < 10.0:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.destroy_subscription(sub)
        if 'urdf' not in holder:
            self.get_logger().error("No /robot_description received. Is the sim/robot running?")
            raise SystemExit(1)
        return holder['urdf']

    def _make_cloud(self, pts):
        # color by height (z) so the reachable shell reads clearly in RViz
        z = pts[:, 2]
        zn = (z - z.min()) / (np.ptp(z) + 1e-9)
        r = (zn * 255).astype(np.uint32)
        b = ((1 - zn) * 255).astype(np.uint32)
        g = np.full_like(r, 80)
        rgb = ((r << 16) | (g << 8) | b).astype(np.uint32)
        # reinterpret the packed uint32 bits as float32 (PCL's rgb convention)
        rgb_f = np.frombuffer(rgb.tobytes(), dtype=np.float32)
        data = [(p[0], p[1], p[2], c) for p, c in zip(pts, rgb_f)]
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        header = self.get_clock().now().to_msg()
        from std_msgs.msg import Header
        h = Header()
        h.stamp = header
        h.frame_id = self.base_frame
        return point_cloud2.create_cloud(h, fields, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot', default='lite6', choices=list(FRAMES))
    ap.add_argument('--samples', type=int, default=30000)
    args = ap.parse_args()

    rclpy.init()
    node = WorkspaceCloud(args.robot, args.samples)
    try:
        rclpy.spin(node)  # keep latching until Ctrl-C
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
