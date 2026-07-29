#!/usr/bin/env python3
"""Spawn printer collision-box models into a running Gazebo, to verify them.

Launch a sim first (any gz world with a /world/<WORLD>/create service — e.g.
scripts/launchVirtualXArm6.sh), then run this. Spawns each model's box body via
a persistent bridged create client (no per-model ros2 process, no TF warmup), so
it's fast (~0.3 s/model after a one-time bridge setup) and consistent.

Config-driven — edit the CONFIG block below (no command-line args).

Notes:
- The A1/A1 mini meshes are modeled Y-up; UPRIGHT rolls them +90 deg about X so
  the heavy base sits on the floor (world +Z up). Each pose z = half the model's
  Y-extent so the base rests at z=0.
- BODY_ONLY spawns just the collision boxes (what MoveIt will use). Set it False
  to also spawn the swinging door + ArUco marker (needs TF; slower).
"""

import sys
import os
import time
import math
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from ar4_automation.simulated3DPrinter import Simulated3DPrinter


def main():
    # ===================== CONFIG (edit these) =====================
    WORLD = 'default'          # gz world name (matches /world/<WORLD>/create)
    BODY_ONLY = True           # True = just collision boxes; False = +door+marker
    UPRIGHT = [math.pi / 2, 0.0, 0.0]   # roll +90deg: model +Y (up) -> world +Z

    # (printer_model, [x, y, z] m). z = half the model's Y-extent -> base on floor:
    #   a1 Y=0.507 -> z=0.254 ; a1_mini Y=0.376 -> z=0.188
    PRINTERS = [
        ('a1',      [0.60,  0.30, 0.254]),
        ('a1_mini', [0.60, -0.30, 0.188]),
    ]
    # ===============================================================

    rclpy.init()
    node = Node('spawn_printers')
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    time.sleep(0.3)

    printers = []
    for name, pos in PRINTERS:
        p = Simulated3DPrinter(
            node=node, pos=pos, orient=UPRIGHT, printer_model=name,
            world_name=WORLD, static_marker_texture=None, use_bad_frame=False)
        body_name = f"printer_{name}"          # distinct name per model
        t0 = time.time()
        if BODY_ONLY:
            ok = p.spawn_body_only(body_name=body_name)
        else:
            ok = p.spawn_fast(body_name=body_name)
        node.get_logger().info(
            f"spawned '{body_name}' ({p.printer_model.name}, "
            f"{len(p.printer_model.boxes)} boxes) at {pos} in {time.time()-t0:.2f}s "
            f"{'OK' if ok else 'FAILED'}")
        printers.append(p)

    node.get_logger().info(
        "Done. Look in Gazebo for the printer bodies. Ctrl-C to exit "
        "(the printers stay spawned).")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
