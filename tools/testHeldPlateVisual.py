#!/usr/bin/env python3
"""
Show the grasped-build-plate visual in Gazebo without running any pickup.
Start the sim first (e.g. scripts/launchVirtualXArm6.sh), run this, then jog
the arm from RViz — the plate box stays glued to the gripper. Type a/d/q at
the prompt to attach / detach / quit.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy

from ar4_automation.runner_common import start_node

# ---- Configuration ----
RUN_SIM       = 1          # Gazebo only — the visual lives in the gz world
ROBOT         = 'xarm6'    # 'ar4' | 'lite6' | 'xarm6'
PRINTER_MODEL = 'a1_mini'  # models/printers/<name>.json supplies plate size + grasp offset
# 6DOF override of the grasp: plate CENTER in the GRASP frame (the gripper's
# link_tcp on the xarm robots — its origin is at the fingertips, so the
# default [0, 0, depth/2] starts the plate's edge right there, clear of the
# gripper body). None = take the model JSON's plate section.
OFFSET_POS    = None       # e.g. [0.0, 0.0, 0.09]
OFFSET_RPY    = None       # e.g. [1.5708, 0.0, 0.0]  (euler XYZ)
# 1 also ATTACHES the plate to the MoveIt planning scene (a collision box
# riding on the grasp frame), so plans account for the held plate; check it in
# RViz's PlanningScene display. 0 = Gazebo visual only.
COLLISIONS    = 1


def main():
    rclpy.init()
    node = start_node(sim=RUN_SIM, robot=ROBOT, collisions=COLLISIONS)

    kwargs = {}
    if OFFSET_POS is not None:
        kwargs['offset_pos'] = OFFSET_POS
    if OFFSET_RPY is not None:
        kwargs['offset_rpy'] = OFFSET_RPY
    node.attach_held_plate(PRINTER_MODEL, **kwargs)

    print("\nheld plate up — jog the arm and watch it follow.")
    print("  a = attach   d = detach   q = quit (detaches first)")
    while rclpy.ok():
        try:
            cmd = input(">> ").strip().lower()
        except EOFError:
            break
        if cmd == 'a':
            node.attach_held_plate(PRINTER_MODEL, **kwargs)
        elif cmd == 'd':
            node.detach_held_plate()
        elif cmd == 'q':
            break
    node.detach_held_plate()


if __name__ == '__main__':
    main()
