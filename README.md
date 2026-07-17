# AR4 3D-Printer Automation

Automates a fleet of Bambu Lab A1 Mini printers with an Annin AR4 robot arm
(ROS 2 + MoveIt2). The arm scans ArUco markers to locate each printer, scrapes
finished prints off the build plate, and cycles the print queue.

## Repository layout

```
runFullAutomationWithScrape.py   Entry points, run these from the repo root.
runScrapePlate.py,
runDoubleTransfer.py,
scanFor2Markers.py, scanFor3Markers.py

printer_config.yaml              Printer credentials (copy the .example file).
print_queue.yaml                 What to print and how many times.

ar4_automation/                  Importable library: printerclass (Bambu MQTT/FTPS
                                 client), printer_automation (robot node),
                                 aruco_detector, pose_reader, camera stack,
                                 simulated3DPrinter, runner_common, and moveit2.py
                                 (patched copy of pymoveit2, do not modify).
calibration/                     Camera/hand-eye calibration tools and outputs.
tools/                           Standalone utilities (servo teleop, gripper,
                                 joint monitor, plotting, object generator).
scripts/                         Shell launchers (virtual robot, calibration,
                                 wifi hotspot for the printers).
gcode/                           Print files (.gcode / .3mf) used by the queue.
models/                          Gazebo models (spawned by file path from the sim code).
data/                            Runtime output: printer_state.json, timing CSVs, logs.
analysis/                        Experiment data and plotting scripts for the paper.
archive/                         Dormant experiments (SLAM, odometry, old robot code).
```

Run entry points from the repo root, e.g. `python3 runScrapePlate.py`.
Scripts in `calibration/`, `tools/`, and `archive/` bootstrap `sys.path`
themselves and can be run from anywhere.

## Testing in Gazebo

Most entry points have a sim switch at the top of the file (`RUN_SIM = 1` in
`runScrapePlate.py` / `runDoubleTransfer.py`, `runVirtual = 1` in the
`scanForNMarkers` scripts). In sim the camera feed comes from the simulated
RGBD camera (`/rgbd_camera/image` + `camera_info`, bridged by
`annin_ar4_gazebo`) and simulated printers are spawned in the scene instead of
loading the hardware save file.

```bash
./scripts/launchVirtualRobot.sh        # Gazebo + MoveIt (wait until loaded)
python3 runScrapePlate.py              # markers 1, 2   (with RUN_SIM = 1)
python3 runDoubleTransfer.py           # markers 0, 1, 2 (with RUN_SIM = 1)
python3 scanFor2Markers.py             # interactive menu (with runVirtual = 1)
python3 scanFor3Markers.py
```

The gripper is disabled in sim (physics instability), and the webcam-specific
settings (90 degree feed rotation, distance-scale correction,
`camera_matrix.npz`) don't apply since intrinsics come from the sim camera's
`camera_info`. `runFullAutomationWithScrape.py` has no sim mode: it drives a
real Bambu printer over MQTT/FTPS, which has no simulated counterpart.

## BambuPrinter client (ar4_automation/printerclass.py)

Local-network control of Bambu printers over MQTT (port 8883, TLS with
self-signed certs) and implicit FTPS (port 990). No cloud or Bambu Studio
needed. The protocols are undocumented, so use at your own risk; cert
verification is disabled and it's meant for LAN use only.

```python
from printerclass import BambuPrinter

printer = BambuPrinter(ip="192.168.1.50", access_code="12345678",
                       serial="01S00A123456789")
printer.connect()
printer.enable_debug_listener()          # print live status updates
printer.upload_file_timeout("part.3mf")  # to the SD card
printer.start_print("part.3mf")
printer.disconnect()
```

Main methods:

- `connect()` / `disconnect()` - MQTT session with a background network loop
- `pause()`, `stop()`, `home()` - basic job control
- `send_gcode(line)` / `send_gcode_file(path)` - raw g-code (file variant is
  only sensible for small files, print big ones from the SD card)
- `blink_light(count)` - identify a printer visually
- `start_print(filename, bed_levelling=False, flow_cali=False, ...)` - start a
  file already on the SD card
- `list_files()`, `upload_file(path)`, `upload_file_timeout(path, timeout=10)` -
  SD card access over FTPS; plain `upload_file` can hang at 100% on some
  firmware, the timeout variant works around that
- `set_on_finish(callback)` - called when a print finishes; the script must
  stay alive (e.g. a sleep loop) for the callback to fire

Implicit FTPS needs a small `ftplib.FTP_TLS` subclass that wraps the socket in
TLS immediately on connect, adapted from
https://gist.github.com/hoogenm/de42e2ef85b38179297a0bba8d60778b.

To run a batch, loop over files: `start_print(f)`, wait for the finish
callback to set a flag, repeat. See the print queue handling in
`runFullAutomationWithScrape.py`.
