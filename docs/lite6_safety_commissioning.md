# Lite 6 safety commissioning

The software checks in this repository supplement the Lite 6 controller,
MoveIt collision checking, guarding and the physical emergency stop. They are
not a safety-rated control system.

## Enable the required xarm_api services

`xarm_ros2` disables the four configuration services by default. Copy
`config/xarm_user_params.yaml` into the `xarm_api/config` directory of the
workspace that is actually sourced, then rebuild that workspace:

```bash
cp ~/ar4Automating3DPrinter/config/xarm_user_params.yaml \
  ~/ar4_ws/src/xarm_ros2/xarm_api/config/xarm_user_params.yaml

cd ~/ar4_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select xarm_msgs xarm_api xarm_moveit_config
source install/local_setup.bash
```

If `xarm_ros2` lives in a different workspace, replace `~/ar4_ws` in both
commands. Do not maintain two sourced xarm installations; verify the selected
one with:

```bash
ros2 pkg prefix xarm_api
ros2 pkg prefix xarm_moveit_config
```

After launching the physical robot, all of these must exist:

```bash
ros2 topic echo /ufactory/robot_states --once
ros2 service type /ufactory/set_collision_sensitivity
ros2 service type /ufactory/set_self_collision_detection
ros2 service type /ufactory/set_reduced_mode
ros2 service type /ufactory/set_reduced_max_tcp_speed
ros2 service type /ufactory/set_reduced_max_joint_speed
ros2 service type /ufactory/open_lite6_gripper
ros2 service type /ufactory/close_lite6_gripper
ros2 action list -t | grep -E 'move_action|execute_trajectory'
```

## What the backend enforces

- Fresh `/joint_states` and `/ufactory/robot_states` telemetry.
- Controller error code zero, controller state ready/running, and MoveIt
  external trajectory mode (`mode=1`).
- Collision sensitivity 3, controller self-collision detection enabled,
  reduced mode enabled, TCP speed limited to
  100 mm/s, and joint speed limited to 0.35 rad/s.
- MoveIt velocity and acceleration scaling of 10% on physical hardware.
- Lite 6 URDF joint limits with a 0.035 rad hard-stop margin.
- A configurable Cartesian workspace and maximum manual joint/jog increments.
- Whole-routine waypoint workspace validation before the first move.
- MoveIt self/environment collision checking on every trajectory segment.
- Immediate trajectory cancellation when the controller reports an error.
- No automatic fault clearing or motor enabling after a collision.
- Pick/place/transfer/scrape use the Lite 6 controller's gripper services and
  fail the routine if the service is missing or reports an error.

## First physical test

1. Remove tools and payloads not represented in the URDF.
2. Clear the workspace; keep one operator at the emergency stop.
3. In UFACTORY Studio, confirm payload, TCP, collision sensitivity, reduced
   limits and the physical safety boundary.
4. Start with a single joint increment under 2 degrees.
5. Test one 5 mm Cartesian jog.
6. Validate marker scanning with no plate in the gripper.
7. Commission each automation waypoint individually at reduced speed.
8. Only then test a complete routine, supervised.

The workspace limits in `ar4_automation/robot_config.py` are starter limits.
Narrow them around the measured printer cell before unattended operation.
