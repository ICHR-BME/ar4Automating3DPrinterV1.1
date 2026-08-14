# Camera and robot calibration

The ChArUco board has 11 x 8 squares, 15 mm square length, 11 mm marker
length, and uses `DICT_4X4_50`. Measure the printed board before calibration;
printing with "fit to page" changes its scale and invalidates every pose.

## xArm wrist-camera hand-eye calibration

Start the physical xArm, MoveIt, RealSense driver, and TF tree first. Keep the
board rigidly fixed in the robot workspace. Then run:

```bash
source /opt/ros/humble/setup.bash
source ~/dev_ws/install/local_setup.bash
cd ~/ar4Automating3DPrinter
python3 calibration/handEyeCalibration.py \
  --robot xarm6 \
  --usb-camera \
  --teach-mode
```

Move the arm only through a commissioned interface. Capture 15-25 views with
the board visible at different positions and, importantly, different wrist
orientations:

- `C`: capture the current synchronized board/robot pose
- `S`: solve, validate, and save the calibration
- `Q` or Escape: exit without saving

The program never commands a trajectory. With `--teach-mode` it changes the
UFACTORY controller to mode 2 so the arm can be guided by hand, then restores
the controller mode observed at startup and state 0 on exit. Support the arm
before enabling teach mode and keep
the emergency stop accessible. A solution is saved to
`calibration/xarm6_hand_eye.json` only if the capture set has sufficient
motion diversity and the fixed-board consistency is at most 10 mm translation
RMSE and 3 degrees rotation RMSE.

On the next backend start, ArUco mapping automatically uses the accepted
`link_eef -> camera` measurement. If the file is absent or invalid it falls
back to the camera TF supplied by the robot description.

For movement-only commissioning while the USB camera is unavailable, start
the GUI backend with `XARM_CAMERA_MODE=disabled`. Vision-dependent commands
remain blocked. Once commissioned, `XARM_CAMERA_MODE=webcam` and
`XARM_CAMERA_INDEX=N` select the physical capture device without probing every
camera during backend startup.
