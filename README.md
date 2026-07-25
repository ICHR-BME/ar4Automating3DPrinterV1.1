# 3D-Printer Automation with a Robot Arm

Tends a small farm of Bambu Lab printers with a robot arm. The arm finds each
printer by its ArUco marker, pulls the finished plate off the bed, scrapes the
print loose against a fixed scraper, puts the plate back and starts the next
job in the queue.

Written for the Annin AR4 (hence the name) but it also drives the UFACTORY
Lite 6 and xArm 6 — pick one with `robot='ar4' | 'lite6' | 'xarm6'`, see
`ar4_automation/robot_config.py`. Everything runs on ROS 2 Jazzy + MoveIt 2,
and most of it can be tried in Gazebo before touching hardware.

## Installation

Linux only. Developed on Ubuntu 24.04 and Linux Mint 22.2.

Install ROS 2 Jazzy following the official instructions, then MoveIt and
Gazebo:

```bash
sudo apt install ros-jazzy-moveit
sudo apt install ros-${ROS_DISTRO}-ros-gz
```

Clone this repo and its neighbours into a workspace:

```bash
mkdir -p ~/ar4_ws/src && cd ~/ar4_ws/src
git clone https://github.com/koghalai123/ar4Automating3DPrinter
git clone https://github.com/koghalai123/ar4_ros_driver
git clone https://github.com/ycheng517/ar4_hand_eye_calibration
git clone https://github.com/AndrejOrsula/pymoveit2
git clone https://github.com/JMU-ROBOTICS-VIVA/ros2_aruco

# only for the UFACTORY arms; the fork carries the sim patches
git clone -b lite6-sim-patches https://github.com/koghalai123/xarm_ros2
```

Pull in the hand-eye calibration dependencies and the rest:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/ar4_ws/src
vcs import . --input ar4_hand_eye_calibration/hand_eye_calibration.repos
sudo apt install ros-jazzy-librealsense2* ros-jazzy-realsense2-*
sudo apt install ros-jazzy-controller-manager ros-jazzy-ros-gz-sim \
     ros-jazzy-ros-gz-bridge ros-jazzy-gz-ros2-control ros-jazzy-ros2-control \
     ros-jazzy-ros2-controllers ros-jazzy-tf-transformations
cd ~/ar4_ws
rosdep install --from-paths . --ignore-src -r -y
colcon build
source install/setup.bash
```

Python packages that don't come from ROS:

```bash
sudo apt install python3-pip
pip install numpy scipy pandas matplotlib opencv-python flask paho-mqtt \
    pyyaml trimesh open3d --break-system-packages
```

Finally copy `printer_config.example.yaml` to `printer_config.yaml` and fill in
each printer's IP, access code and serial (all under Settings → Network /
Device on the printer). The real file is gitignored.

## Bringing up a robot

Start one of these first and wait for MoveIt to finish loading, then run an
entry point in a second terminal.

```bash
./scripts/launchVirtualRobot.sh        # AR4 in Gazebo
./scripts/launchVirtualXArm6.sh        # xArm 6 in Gazebo
./scripts/launchVirtualXArmLite6.sh    # Lite 6 in Gazebo
./scripts/launchPhysicalXArm6.sh       # real xArm 6
./scripts/launchPhysicalXArmLite6.sh   # real Lite 6
./scripts/launchCalibrationPhysical.sh # real AR4, homes the joints first
```

`scripts/start_hotspot.sh` brings up the wifi hotspot the printers join.

## Main scripts

Run these from the repo root. Options live in a config block at the top of
each file — there are no command-line flags.

| Script | What it does |
| --- | --- |
| `runFullAutomationWithScrape.py` | The full loop: print a job from `print_queue.yaml`, wait for it, scrape the plate, repeat. Real printers only. |
| `runScrapePlate.py` | One cycle: take the plate off one printer, scrape it against another, put it back. |
| `runPickupPlate.py` | Just the pickup, then stop. Useful when tuning approach offsets. |
| `runDoubleTransfer.py` | Shuttles plates between three stations in a loop. |
| `scanFor2Markers.py`, `scanFor3Markers.py` | Locate the markers, then drop into an interactive menu of moves. This is the usual starting point. |
| `teachMarkersByHand.py` | Puts a UFACTORY arm in drag-teach mode so you can walk the camera to each marker by hand instead of typing coordinates. Saves to `data/manual_marker_estimates.json`. |
| `runCalibrateCameraOffset.py` | Orbits a stationary marker to measure the end-effector-to-camera mount error. |
| `xArm6LiteControl.py` | Minimal joint/pose commands for a Lite 6, handy for checking a new setup. |
| `robot_agent.py` | HTTP server exposing a whitelist of the above to the web dashboard. See `robot_link.py` for the protocol. |

Most of the motion scripts have a `RUN_SIM` (or `runVirtual`) switch at the top
for Gazebo. In sim the camera feed comes from the simulated RGBD camera and
fake printers are spawned in the scene, so the webcam settings — 90° feed
rotation, distance correction, `camera_matrix.npz` — don't apply. The gripper
is disabled in sim because the physics are unstable.
`runFullAutomationWithScrape.py` has no sim mode: it talks to a real printer
over MQTT.

## Layout

```
ar4_automation/     The library. printer_automation.py is the robot node,
                    printerclass.py the Bambu client, plus aruco_detector,
                    pose_reader, the camera stack, simulated3DPrinter,
                    robot_config and runner_common. moveit2.py is a patched
                    copy of pymoveit2 — don't edit it.
calibration/        Camera intrinsics, hand-eye calibration, marker PDFs.
tools/              Standalone utilities: servo teleop, gripper cycling, live
                    joint readout, workspace point cloud, plotting.
scripts/            Launch wrappers and the printer hotspot.
gcode/              Print files used by the queue.
models/             Gazebo models, spawned by file path from the sim code.
data/               Runtime output: saved marker poses, timing CSVs, logs.
dataAnalysis/       Experiment data and the plotting scripts for the paper.
archive/            Dormant experiments (SLAM, odometry, old robot code).
```

Scripts under `calibration/`, `tools/` and `archive/` set up `sys.path`
themselves and can be run from anywhere.

## Talking to the printers

`ar4_automation/printerclass.py` controls Bambu printers over the LAN — MQTT
on 8883 for status and commands, implicit FTPS on 990 for the SD card. No
cloud account or Bambu Studio involved. The protocols are undocumented and
certificate verification is off, so keep it on a network you trust.

```python
from printerclass import BambuPrinter

printer = BambuPrinter(ip="192.168.1.50", access_code="12345678",
                       serial="01S00A123456789")
printer.connect()
printer.enable_debug_listener()          # live status updates
printer.upload_file_timeout("part.3mf")  # onto the SD card
printer.start_print("part.3mf")
printer.disconnect()
```

Worth knowing:

- `pause()`, `stop()`, `home()` — basic job control.
- `send_gcode(line)` / `send_gcode_file(path)` — raw g-code. Only sensible for
  short files; print anything real from the SD card.
- `list_files()`, `upload_file()`, `upload_file_timeout()` — plain
  `upload_file` can hang at 100% on some firmware, so prefer the timeout one.
- `set_on_finish(callback)` — fires when a print completes, but only while the
  script is still alive. `runFullAutomationWithScrape.py` shows the pattern.
- `blink_light(count)` — flashes the chamber light, for working out which
  printer is which.

Implicit FTPS needs an `ftplib.FTP_TLS` subclass that wraps the socket in TLS
on connect, adapted from
<https://gist.github.com/hoogenm/de42e2ef85b38179297a0bba8d60778b>.
