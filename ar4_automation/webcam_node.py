import csv
import os
import cv2
import subprocess
import re
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


def list_cameras():
    """List available video devices with their names."""
    cameras = []
    try:
        result = subprocess.run(['ls', '/sys/class/video4linux/'], capture_output=True, text=True)
        devices = sorted(result.stdout.strip().split())
        for device in devices:
            index = int(re.search(r'\d+', device).group())
            try:
                with open(f'/sys/class/video4linux/{device}/name', 'r') as f:
                    name = f.read().strip()
            except FileNotFoundError:
                name = "Unknown"
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cameras.append((index, name))
                cap.release()
    except Exception as e:
        print(f"Warning: Could not enumerate devices: {e}")
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append((i, f"Camera {i}"))
                cap.release()
    return cameras


def get_camera_usb_properties(camera_index: int) -> dict:
    """Query USB/device properties via udevadm and sysfs."""
    device_path = f"/dev/video{camera_index}"
    properties = {"device_path": device_path, "camera_index": str(camera_index)}

    try:
        result = subprocess.run(
            ['udevadm', 'info', '--query=all', f'--name={device_path}'],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if line.startswith('E:'):
                key, _, value = line[3:].partition('=')
                properties[key.strip()] = value.strip()
    except FileNotFoundError:
        print("Warning: udevadm not found; skipping udev property query.")
    except Exception as e:
        print(f"Warning: udevadm query failed: {e}")

    # sysfs fills in anything udevadm missed
    usb_attrs = [
        'manufacturer', 'product', 'serial',
        'idVendor', 'idProduct', 'bcdDevice',
        'speed', 'version', 'busnum', 'devnum',
    ]
    try:
        device_link = f"/sys/class/video4linux/video{camera_index}/device"
        path = os.path.realpath(device_link)
        for _ in range(6):  # walk up toward the USB root, 6 levels max
            for attr in usb_attrs:
                attr_file = os.path.join(path, attr)
                if os.path.isfile(attr_file):
                    key = f"usb_{attr}"
                    if key not in properties:  # udevadm values win
                        with open(attr_file) as f:
                            properties[key] = f.read().strip()
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent
    except Exception as e:
        print(f"Warning: Could not read sysfs USB attributes: {e}")

    return properties


def write_camera_properties_to_csv(properties: dict, filename: str = "camera_properties.csv"):
    """Dump properties as key-value rows to CSV."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["property", "value"])
        for key, value in properties.items():
            writer.writerow([key, value])
    print(f"Camera properties saved to '{filename}'")


def select_camera(preset_keyword=None):
    """List cameras and let the user choose, or auto-select by keyword."""
    cameras = list_cameras()

    if not cameras:
        print("No cameras found!")
        return None

    if preset_keyword is not None:
        keyword_lower = preset_keyword.lower()
        for idx, name in cameras:
            if keyword_lower in name.lower():
                print(f"Auto-selected camera [{idx}] '{name}' (matched keyword '{preset_keyword}')")
                return idx
        print(f"Warning: No camera found matching keyword '{preset_keyword}'")
        print("Falling back to manual selection...\n")

    print("\nAvailable cameras:")
    print("-" * 40)
    for idx, name in cameras:
        print(f"  [{idx}] {name}")
    print("-" * 40)

    while True:
        try:
            choice = int(input("Select camera index: "))
            if choice in [c[0] for c in cameras]:
                return choice
            else:
                print(f"Invalid choice. Available indices: {[c[0] for c in cameras]}")
        except ValueError:
            print("Please enter a valid integer.")


class WebcamPublisher(Node):
    def __init__(
        self,
        camera_keyword: str = "GENERAL WEBCAM",
        show_preview: bool = False,
        camera_matrix: np.ndarray = None,
        dist_coeffs: np.ndarray = None,
        publish_rate: float = 30.0,
        image_topic: str = '/rgbd_camera/image',
        depth_topic: str = '/rgbd_camera/depth_image',
        camera_info_topic: str = '/rgbd_camera/camera_info',
        frame_id: str = 'ee_camera_link',
    ):
        super().__init__('webcam_publisher')
        self.bridge = CvBridge()
        self.show_preview = show_preview
        self.frame_id = frame_id

        camera_index = select_camera(preset_keyword=camera_keyword)
        if camera_index is None:
            self.get_logger().error("No camera selected")
            raise RuntimeError("No camera selected")

        self.color_pub = self.create_publisher(Image, image_topic, 10)
        self.depth_pub = self.create_publisher(Image, depth_topic, 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, camera_info_topic, 10)

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open camera at index {camera_index}")
            raise RuntimeError(f"Could not open camera at index {camera_index}")

        # grab one frame to learn the resolution
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Could not read initial frame from camera")
            raise RuntimeError("Could not read initial frame from camera")

        self.img_height, self.img_width = frame.shape[:2]
        self.get_logger().info(f"Camera opened: {self.img_width}x{self.img_height} (index {camera_index})")

        usb_props = get_camera_usb_properties(camera_index)
        usb_props["resolution_width"] = str(self.img_width)
        usb_props["resolution_height"] = str(self.img_height)
        csv_filename = f"camera_{camera_index}_properties.csv"
        write_camera_properties_to_csv(usb_props, csv_filename)
        self.get_logger().info(f"USB device properties written to '{csv_filename}'")

        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        else:
            fx = fy = float(self.img_width)
            cx = self.img_width / 2.0
            cy = self.img_height / 2.0
            self.camera_matrix = np.array([
                [fx,  0.0, cx],
                [0.0, fy,  cy],
                [0.0, 0.0, 1.0]
            ])
            self.dist_coeffs = np.zeros(5)
            self.get_logger().warn(
                f"Using estimated camera intrinsics (fx={fx}, fy={fy}, cx={cx}, cy={cy}). "
                "Pass camera_matrix for accurate pose estimation."
            )

        self.camera_info_msg = self._build_camera_info()

        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)
        self.get_logger().info(
            f"Publishing on {image_topic}, {depth_topic}, "
            f"{camera_info_topic} at {publish_rate} Hz"
        )

    def _build_camera_info(self) -> CameraInfo:
        msg = CameraInfo()
        msg.header.frame_id = self.frame_id
        msg.width = self.img_width
        msg.height = self.img_height
        msg.distortion_model = "plumb_bob"
        msg.d = self.dist_coeffs.flatten().tolist()
        msg.k = self.camera_matrix.flatten().tolist()
        msg.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        msg.p = [fx, 0.0, cx, 0.0,
                  0.0, fy, cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        return msg

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame from camera")
            return

        stamp = self.get_clock().now().to_msg()

        color_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        color_msg.header.stamp = stamp
        color_msg.header.frame_id = self.frame_id
        self.color_pub.publish(color_msg)

        # dummy all-zeros depth, downstream expects the topic to exist
        depth_dummy = np.zeros((self.img_height, self.img_width), dtype=np.float32)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_dummy, encoding="32FC1")
        depth_msg.header.stamp = stamp
        depth_msg.header.frame_id = self.frame_id
        self.depth_pub.publish(depth_msg)

        # Publish camera info
        self.camera_info_msg.header.stamp = stamp
        self.camera_info_pub.publish(self.camera_info_msg)

        # Optionally show a local preview
        if self.show_preview:
            cv2.imshow("Webcam Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Quit key pressed, shutting down...")
                rclpy.shutdown()

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        if self.show_preview:
            cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    try:
        node = WebcamPublisher(
            camera_keyword="GENERAL WEBCAM",
            show_preview=True,
            publish_rate=30.0,
        )
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\n{e}" if str(e) else "\nShutting down...")
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()