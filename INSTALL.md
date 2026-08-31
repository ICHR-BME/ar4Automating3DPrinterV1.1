# Installation — xArm6 / Lite6 branch (`integration/lite6-robot-control`)

Verified end to end on **Ubuntu 22.04 + ROS 2 Humble** against a physical
xArm6 controller, August 2026.

The `main` branch targets ROS 2 Jazzy on Ubuntu 24.04. **This branch runs on
Humble.** Do not follow `main`'s README on a 22.04 machine.

> This fork (`ICHR-BME/ar4Automating3DPrinterV1.1`) tracks
> `koghalai123/ar4Automating3DPrinter` and adds the OpenCV 4.7+ ArUco fix
> and this document. Upstream does not yet contain either.

Two repositories are involved:

| Repo | Purpose | Location used here |
|---|---|---|
| `UCDavisGUI` | FastAPI + React web GUI | `~/UCDavisGUI` |
| `ar4Automating3DPrinter` | robot/vision library imported by the GUI | `~/ar4Automating3DPrinter` |

A third-party ROS workspace is built separately at `~/xarm_ws`.

---

## 1. ROS 2 Humble

Humble requires Jammy. Confirm before starting:

```bash
lsb_release -cs        # must print: jammy
```

```bash
sudo apt update && sudo apt install -y \
  curl gnupg lsb-release software-properties-common git python3-pip python3-venv
sudo add-apt-repository universe -y

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop ros-dev-tools python3-rosdep

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

sudo rosdep init      # harmless if it says it already exists
rosdep update
```

Verify:

```bash
python3 -c "import rclpy; print('rclpy ok')"
```

Node.js 20 (for the GUI frontend build):

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node --version        # v20.x
```

---

## 2. ROS system packages

`rosdep` covers most of these, but installing them up front avoids a failed
first build:

```bash
sudo apt install -y \
  ros-humble-moveit \
  ros-humble-moveit-msgs \
  ros-humble-controller-manager \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-joint-state-broadcaster \
  ros-humble-joint-trajectory-controller \
  ros-humble-tf-transformations \
  ros-humble-urdfdom-py \
  ros-humble-vision-msgs \
  ros-humble-ros-gz-interfaces \
  python3-transforms3d
```

`ros-humble-ros-gz-interfaces` is **required even for hardware-only use.**
`ar4_automation/simulated3DPrinter.py` imports `ros_gz_interfaces` at module
level, and that module is pulled in transitively by `runner_common`, so the
import runs whether or not you ever use simulation.

---

## 3. xArm workspace

```bash
mkdir -p ~/xarm_ws/src && cd ~/xarm_ws/src
git clone -b humble --recursive https://github.com/xArm-Developer/xarm_ros2.git
git clone https://github.com/AndrejOrsula/pymoveit2.git
```

`pymoveit2` is a colcon package, not a pip package — `pip install pymoveit2`
does not exist. It is required: `printer_automation.py` imports
`GripperInterface` and `xArm6LiteControl.py` imports `MoveIt2` from it.

### 3a. Enable the safety services (required)

`pose_reader.py` applies a UFACTORY safety profile on startup and refuses to
unblock motion if any of five services is missing. All five ship **disabled**
in `xarm_ros2`. Without this edit the backend logs:

```
UFACTORY safety profile was not applied; motion will remain blocked:
service unavailable: /xarm/set_collision_sensitivity
```

```bash
cd ~/xarm_ws/src/xarm_ros2/xarm_api/config
cp xarm_params.yaml xarm_params.yaml.bak
sed -i \
 -e 's/^\( *\)set_collision_sensitivity: false/\1set_collision_sensitivity: true/' \
 -e 's/^\( *\)set_self_collision_detection: false/\1set_self_collision_detection: true/' \
 -e 's/^\( *\)set_reduced_max_tcp_speed: false/\1set_reduced_max_tcp_speed: true/' \
 -e 's/^\( *\)set_reduced_max_joint_speed: false/\1set_reduced_max_joint_speed: true/' \
 -e 's/^\( *\)set_reduced_mode: false/\1set_reduced_mode: true/' \
 xarm_params.yaml
grep -n "set_collision_sensitivity\|set_self_collision_detection\|set_reduced" xarm_params.yaml
```

All five must read `true`.

### 3b. Build

**Deactivate any Python virtualenv first.** If a venv is active, CMake picks up
its interpreter, which has no `catkin_pkg`, and every package fails with a
misleading `ModuleNotFoundError`. If you have already hit this, delete
`build/ install/ log/` before retrying — CMake caches the wrong interpreter
path and reuses it.

```bash
deactivate 2>/dev/null
cd ~/xarm_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
```

About 3 minutes for 14 packages. Then:

```bash
echo "source ~/xarm_ws/install/setup.bash" >> ~/.bashrc
```

---

## 4. Clone the repositories

```bash
git clone https://github.com/adrianSalazar050/UCDavisGUI ~/UCDavisGUI
git clone https://github.com/ICHR-BME/ar4Automating3DPrinterV1.1 ~/ar4Automating3DPrinter
cd ~/ar4Automating3DPrinter
git checkout integration/lite6-robot-control
```

`ar4Automating3DPrinter` is not a colcon package. The GUI adds it to
`sys.path` and imports `ar4_automation` directly, so it stays outside
`~/xarm_ws`.

---

## 5. Python environment (GUI)

```bash
cd ~/UCDavisGUI
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install scipy open3d trimesh flask pandas
```

`python3 -m venv` does **not** overwrite an existing `.venv` directory. If one
is already present it is silently reused. Delete it first if you want a clean
environment.

Packages missing from `requirements.txt` but imported at runtime:

| Package | Imported by |
|---|---|
| `scipy` | `server/robot.py` (`jog_pose`), `pose_reader.py` |
| `open3d`, `trimesh` | `ar4_automation` geometry helpers |
| `flask` | `ar4_automation/web_video_server.py` |
| `pandas` | analysis helpers |

Verify the NumPy pin held — ROS 2 Humble's `cv_bridge` and `transforms3d` are
built against the NumPy 1.x ABI and fail to import against 2.x:

```bash
pip show numpy | grep "^Version:"     # must be 1.26.x
```

If `scipy` is installed before `requirements.txt`, it pulls in NumPy 2.x.
Installing `requirements.txt` afterwards downgrades it correctly, but check.

### Frontend

```bash
cd ~/UCDavisGUI/frontend
npm install
npm run build
cd ..
```

Mandatory. Without `frontend/dist/index.html` the server serves a plain-text
placeholder instead of the GUI.

### Verify

```bash
cd ~/UCDavisGUI
source /opt/ros/humble/setup.bash
source ~/xarm_ws/install/setup.bash
source .venv/bin/activate        # ROS first, venv last

python3 -c "
import rclpy, xarm_msgs, pymoveit2, cv_bridge, numpy, cv2
print('imports ok', numpy.__version__, cv2.__version__)"

python3 -c "
import sys; sys.path.insert(0, '$HOME/ar4Automating3DPrinter')
import ar4_automation.runner_common; print('runner_common ok')"
```

Both must pass before going further.

---

## 6. Running

Three terminals. Each needs both ROS setups sourced.

### Terminal 1 — driver and MoveIt

Find the controller if you do not know its address:

```bash
sudo apt install -y arp-scan
sudo arp-scan --interface=<iface> --localnet
ping -c 3 <robot-ip>
nc -zv -w 3 <robot-ip> 502        # UFACTORY control port
```

```bash
source /opt/ros/humble/setup.bash
source ~/xarm_ws/install/setup.bash
ros2 launch xarm_moveit_config xarm6_moveit_realmove.launch.py robot_ip:=<robot-ip>
```

Wait for `You can start planning now!`.

### Terminal 2 — confirm the arm is live

```bash
ros2 topic hz /joint_states           # expect a steady ~10 Hz
ros2 service list | grep -E "collision|reduced"   # expect all five
```

Do not start the GUI until both pass.

### Terminal 3 — GUI backend

```bash
cd ~/UCDavisGUI
source /opt/ros/humble/setup.bash
source ~/xarm_ws/install/setup.bash
source .venv/bin/activate

python -m server --robot-mode ros --robot-type xarm6 \
  --robot-repo ~/ar4Automating3DPrinter
```

Do **not** pass `--robot-sim`; that targets Gazebo, not the physical arm.

Open <http://127.0.0.1:8000> and go to **Robot**.

Optional environment variables (documented in `calibration/README.md`):

```bash
export XARM_CAMERA_MODE=webcam    # or 'disabled' for movement-only commissioning
export XARM_CAMERA_INDEX=2        # skips probing every camera at startup
```

### Camera index

On a laptop, `/dev/video0` is usually the built-in webcam. Identify the right
one before selecting it in the GUI:

```bash
sudo apt install -y v4l-utils
v4l2-ctl --list-devices
```

Use the index of the external camera (`/dev/video2` → index `2`), not `0`.

---

## 7. Hand-eye calibration — required for vision commands

ArUco detection works as soon as the camera is selected: markers are found and
pose axes are drawn in the preview. **Mapping those markers into robot base
coordinates does not**, unless a hand-eye calibration exists.

Without it the backend logs, once per frame:

```
[enrich] ID=n cameraToBase FAILED — TF lookup returned None.
found_markers will NOT be updated with real pose.
```

and `usb_camera_optical_frame` is absent from the TF tree
(`ros2 run tf2_ros tf2_monitor` lists only `link1`–`link6`, `link_base`,
`link_eef`).

The expected result file is `calibration/xarm6_hand_eye.json`. **It is not in
the repository.** Until it is generated, `scan_marker`, `pickup`, `place`,
`transfer`, and `scrape` cannot complete.

See `calibration/README.md` for the procedure. Two things to check first:

- Print `charuco_board.pdf` at **100% scale** and measure it. "Fit to page"
  silently invalidates every pose.
- Calibration encodes the physical camera mounting. If the mount is moved or
  reoriented afterwards, the calibration must be redone. Fix any mounting
  work **before** calibrating.

---

## Troubleshooting

**`colcon build` fails with `No module named 'catkin_pkg'`**
A virtualenv was active. `deactivate`, delete `build/ install/ log/`, rebuild.

**Every `apt install` fails with unmet `libc6 (>= 2.38)` / `libelf1t64`**
A mainline or 24.04-targeted kernel package is half-installed and jamming apt.
`sudo apt --fix-broken install`, then purge the offending
`linux-headers-*` / `linux-image-*` packages. Check `uname -r` first and never
remove the running kernel.

**`pip install` fails with `Could not find a version that satisfies …`**
Usually an unquoted version specifier on the command line — `>=` is a shell
redirect. Quote it: `pip install "paho-mqtt>=2.0"`.

**Camera frames are half image, half flat grey**
Pixel-format negotiation. Confirm what the device offers with
`v4l2-ctl -d /dev/videoN --list-formats-ext` and what it negotiated with
`--get-fmt-video`. MJPG at 1280x720 works.

**`module 'cv2.aruco' has no attribute 'detectMarkers'`**
Fixed on this branch. Do not "solve" it by downgrading to OpenCV 4.6 —
`cv2.aruco` does not exist in `opencv-python` below 4.7 (it lived in
`opencv-contrib-python`), and 4.6 also conflicts with `ultralytics`.

**`Waiting for joint_states…` / `service unavailable: /xarm/set_collision_sensitivity`**
The driver launch is not running, or step 3a was skipped.

**Motion controls stay greyed out**
The GUI enables them only when `robot.available` and `safety.ready` are both
true. `safety.ready` depends on the five services from step 3a.

**Gripper buttons do nothing**
`xarm6_moveit_realmove.launch.py` launches with `add_gripper: 0`. The backend
logs `No gripper configured for robot 'xarm6'`, and `pickup` / `place` are
skipped regardless of camera state.

**`FATAL: exception not rethrown` / `Aborted (core dumped)` on Ctrl-C**
Cosmetic rclpy shutdown ordering. Ignore.
