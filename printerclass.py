import paho.mqtt.client as mqtt
import json
import ssl
import ftplib
import socket
import struct
import os
import sys
import time
import yaml

# ==========================================
# Helper Classes
# ==========================================

class ImplicitFTP_TLS(ftplib.FTP_TLS):
    """
    FTP_TLS subclass to support implicit FTPS.
    Source: https://gist.github.com/hoogenm/de42e2ef85b38179297a0bba8d60778b
    Implicit FTPS requires the connection to be wrapped in SSL immediately upon connection.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sock = None

    @property
    def sock(self):
        return self._sock

    @sock.setter
    def sock(self, value):
        if value is not None and not isinstance(value, ssl.SSLSocket):
            value = self.context.wrap_socket(value)
        self._sock = value

# ==========================================
# Main Printer Class
# ==========================================

class BambuPrinter:
    def __init__(self, ip, access_code, serial):
        """
        Initializes the connection parameters for the Bambu Printer.
        :param ip: Local IP address of the printer '192.xxxxxxx '
        :param access_code: 8-digit access code found in printer settings 'xxxxxx'
        :param serial: Printer serial number 'xxAbeasax12da32xxxxxx'
        """
        self.ip = ip
        self.access_code = access_code
        self.serial = serial
        self.topic_publish = f"device/{self.serial}/request"
        self.topic_subscribe = f"device/{self.serial}/report"
        
        # Setup the MQTT Client
        self.client = mqtt.Client()
        self.client.username_pw_set("bblp", self.access_code)
        self.on_finish_callback = None
        self._last_state = None

        # Live status panel: latest known values, redrawn in place on one line.
        self.debug = False          # set True to dump raw MQTT payloads
        self.status = {}            # e.g. {"state": ..., "progress": ..., "nozzle": ...}
        
        # SSL configuration for MQTT (Bambu uses self-signed certs)
        self._print_finished = False

        self.client.tls_set(cert_reqs=ssl.CERT_NONE)
        self.client.tls_insecure_set(True)

        # Setup FTP SSL Context for file transfers
        self.ftp_context = ssl.create_default_context()
        self.ftp_context.check_hostname = False
        self.ftp_context.verify_mode = ssl.CERT_NONE
        self.ftp_context.options |= ssl.OP_IGNORE_UNEXPECTED_EOF

    # ------------------------------------------
    # MQTT Communication Methods
    # ------------------------------------------

    def connect(self):
        """Establishes connection to the printer's MQTT broker.

        Tries the configured IP first; if the printer isn't reachable there
        (e.g. DHCP handed it a new address), searches the local network by
        serial via SSDP and uses the discovered IP instead.
        """
        if not self._is_reachable(self.ip):
            print(f"Printer not reachable at {self.ip}; searching the local network...")
            found_ip = self.discover_ip()
            if found_ip:
                print(f"Found printer {self.serial} at {found_ip} (was {self.ip}).")
                self.ip = found_ip
            else:
                print(f"Warning: printer {self.serial} not found on the network; "
                      f"trying {self.ip} anyway.")

        self.client.connect(self.ip, 8883, 60)
        self.client.loop_start()  # Starts background thread for networking
        print(f"Connected to printer {self.serial}")

    def disconnect(self):
        """Stops the background loop and disconnects MQTT."""
        self.client.loop_stop()
        self.client.disconnect()

    def _is_reachable(self, ip, port=8883, timeout=3.0):
        """Return True if a TCP connection to ip:port (the MQTT port) succeeds."""
        if not ip:
            return False
        try:
            with socket.create_connection((ip, port), timeout=timeout):
                return True
        except OSError:
            return False

    def discover_ip(self, serial=None, timeout=10.0):
        """Find the printer's IP on the LAN via Bambu's SSDP broadcasts.

        Bambu printers periodically multicast SSDP NOTIFY packets to
        239.255.255.250 (UDP 2021/1990) carrying their serial (USN header) and
        IP (Location header / source address). Listens up to *timeout* seconds
        and returns the IP whose USN matches *serial* (defaults to this
        printer's serial), or None if it isn't seen in time.
        """
        serial = serial or self.serial
        mcast_grp = "239.255.255.250"

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass

        bound = False
        for port in (2021, 1990):  # Bambu broadcasts on both; share via SO_REUSE*
            try:
                sock.bind(("", port))
                bound = True
                break
            except OSError:
                continue
        if not bound:
            print("Discovery: could not bind SSDP port (2021/1990).")
            sock.close()
            return None

        # Join the SSDP multicast group on all interfaces.
        mreq = struct.pack("4s4s", socket.inet_aton(mcast_grp), socket.inet_aton("0.0.0.0"))
        try:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except OSError:
            pass  # may still receive on the bound port without an explicit join

        sock.settimeout(1.0)
        print(f"Searching local network for printer {serial} (up to {timeout:.0f}s)...")
        deadline = time.time() + timeout
        seen = {}
        try:
            while time.time() < deadline:
                try:
                    data, addr = sock.recvfrom(2048)
                except socket.timeout:
                    continue
                except OSError:
                    break

                text = data.decode(errors="ignore")
                if "USN" not in text and "bambu" not in text.lower():
                    continue

                headers = {}
                for line in text.split("\r\n"):
                    key, sep, val = line.partition(":")
                    if sep:
                        headers[key.strip().lower()] = val.strip()

                usn = headers.get("usn", "")
                ip = headers.get("location", "") or addr[0]
                if usn:
                    seen[usn] = ip
                if serial and serial in usn:
                    return ip
        finally:
            sock.close()

        if seen:
            others = ", ".join(f"{s}@{i}" for s, i in seen.items())
            print(f"Discovery: printer {serial} not found. Other Bambu devices seen: {others}")
        return None

    def enable_debug_listener(self):
        """Sets the callback and subscribes without blocking the script."""
        def on_message(client, userdata, msg):
            payload = json.loads(msg.payload.decode())
            # Debug: dump command acks (e.g. the response to project_file)
            if self.debug and "print" in payload and payload["print"].get("command") in ("project_file", "push_status"):
                print("\nRAW:", json.dumps(payload, indent=2))
            self._parse_message(payload)
        
        self.client.on_message = on_message
        self.client.subscribe(self.topic_subscribe)
        print(f"Listener active for {self.serial}")

    def set_on_finish(self, func):
        """Pass a function here to be called when the print finishes."""
        self.on_finish_callback = func

    # Maps incoming report fields -> (status key, formatter)
    _STATUS_FIELDS = [
        ("gcode_state",  "state",    lambda v: v),
        ("mc_percent",   "progress", lambda v: f"{v}%"),
        ("layer_num",    "layer",    lambda v: v),
        ("nozzle_temper", "nozzle",  lambda v: f"{round(v, 1)}°C"),
        ("bed_temper",   "bed",      lambda v: f"{round(v, 1)}°C"),
    ]

    def _parse_message(self, data):
        """Update the live status values from a report and redraw the panel."""
        print_data = data.get("print", {})
        if not print_data:
            return

        changed = False
        for src_key, status_key, fmt in self._STATUS_FIELDS:
            if src_key in print_data:
                value = fmt(print_data[src_key])
                if self.status.get(status_key) != value:
                    self.status[status_key] = value
                    changed = True

        if changed:
            self._render_status()

        # Finish detection (no longer spams the log on every state change)
        new_state = print_data.get("gcode_state")
        if new_state and new_state != self._last_state:
            self._last_state = new_state
            if new_state == "FINISH":
                self._print_finished = True
                if self.on_finish_callback:
                    self.on_finish_callback()

    def _render_status(self):
        """Redraw all known status values in place on a single line."""
        order = ["state", "progress", "layer", "nozzle", "bed"]
        labels = {"state": "State", "progress": "Progress",
                  "layer": "Layer", "nozzle": "Nozzle", "bed": "Bed"}
        parts = [f"{labels[k]}: {self.status[k]}" for k in order if k in self.status]
        # \r returns to line start, \033[K clears to end of line (handles shrinking text)
        sys.stdout.write("\r\033[K" + "  |  ".join(parts))
        sys.stdout.flush()


    def waitUntilPrintFinished(self, poll_interval=1.0):
        """Blocks until the printer reports a FINISH state via MQTT."""
        print("Waiting for print to finish...")
        self._print_finished = False
        while not self._print_finished:
            time.sleep(poll_interval)
        sys.stdout.write("\n")  # preserve the final status line
        print("Print finished. Continuing.")

    def _send_command(self, command_dict):
        """Internal helper to package and send JSON payloads over MQTT."""
        payload = json.dumps(command_dict)
        self.client.publish(self.topic_publish, payload)

    # ------------------------------------------
    # Printer Control Commands
    # ------------------------------------------

    def pause(self):
        """Pauses the current print job."""
        cmd = {"print": {"sequence_id": "0", "command": "pause"}}
        self._send_command(cmd)
        print("Command sent: Pause")

    def stop(self):
        """Aborts the current print job."""
        cmd = {"print": {"sequence_id": "0", "command": "stop"}}
        self._send_command(cmd)
        print("Command sent: Stop")

    def home(self):
        """Moves all axes to the home position (X0 Y0 Z0)."""
        self.send_gcode("G0 X0 Y0 Z0 F1200")
        print("Command sent: Move to home position")

    def homing(self):
        """Finds axis limits via G28, then moves to the home position."""
        self.send_gcode("G28")
        print("Command sent: Homing (finding limits)")
        time.sleep(10)  # Wait for homing to complete
        self.home()

    def prepare_for_pickup(self):
        """Move the tool head to max X/Y/Z so the build plate is accessible for the robot arm."""
        self.send_gcode("G0 X180 Y180 Z180 F1200")

        time.sleep(10)
        print("Printer ready for pickup")

    def send_gcode(self, gcode_line):
        """Sends a single line of raw G-code to the printer."""
        cmd = {
            "print": {
                "sequence_id": "0",
                "command": "gcode_line",
                "param": f"{gcode_line}"
            }
        }
        self._send_command(cmd)

    def blink_light(self, count=10): 
        """Blinks the chamber light for debugging/identification purposes."""
        for idx in range(1, count):
            mode = "on" if idx % 2 == 0 else "off"
            cmd = {"system": {"command": "ledctrl", "led_node": "chamber_light", "led_mode": mode}}
            self._send_command(cmd)
            time.sleep(0.25)
    
    def send_gcode_file(self, gcode_filename):
        """Reads a local file and sends its contents as a G-code command block."""
        gcode_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), gcode_filename)
        with open(gcode_filename, 'r') as f:
            data = f.read()
            self.send_gcode(data)
    
    def start_print(self, filename, bed_levelling=False, flow_cali=False, vibration_cali=False, use_ams=False):
        """
        Triggers the printer to start a print from a file already on the SD card.
        """
        file_url = f"ftp://{filename}"
        
        cmd = {
            "print": {
                "sequence_id": "0", 
                "command": "project_file",
                "param": "Metadata/plate_1.gcode",
                "subtask_name": filename,
                "url": file_url,
                "timelapse": False,
                "bed_leveling": bed_levelling,
                "flow_cali": flow_cali,
                "vibration_cali": vibration_cali,
                "layer_inspect": True,
                "use_ams": use_ams
            }
        }
        
        print(f"Sending Start Command: {filename}")
        self._send_command(cmd)

    # ------------------------------------------
    # FTP / File Management Methods
    # ------------------------------------------

    def list_files(self):
        """Connects via FTPS and lists files on the SD card."""
        try:
            with ImplicitFTP_TLS(context=self.ftp_context) as ftps:
                ftps.connect(host=self.ip, port=990)
                ftps.login(user="bblp", passwd=self.access_code)
                ftps.prot_p()
                print(f"--- Files on {self.serial} SD Card ---")
                ftps.retrlines('LIST')
        except Exception as e:
            print(f"FTP Error: {e}")

    def upload_file(self, local_path):
        """
        Uploads a file to the SD card via FTPS.
        Note: Known issue where some transfers may hang at 100%.
        """
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), local_path)
        filename = os.path.basename(local_path)
        file_size = os.path.getsize(local_path)
        self.bytes_sent = 0

        def progress_callback(data):
            self.bytes_sent += len(data)
            percent = (self.bytes_sent / file_size) * 100
            sys.stdout.write(
                f"\rUploading {filename}: {percent:.1f}% "
                f"({self.bytes_sent}/{file_size} bytes)"
            )
            sys.stdout.flush()

        try:
            with ImplicitFTP_TLS(context=self.ftp_context) as ftps:
                ftps.connect(host=self.ip, port=990)
                ftps.login(user="bblp", passwd=self.access_code)
                ftps.prot_p()
                ftps.set_pasv(True)
               
                with open(local_path, 'rb') as f:
                    print(f"Starting upload: {filename}")
                    ftps.set_debuglevel(2)
                    ftps.storbinary(f"STOR {filename}", f, callback=progress_callback)

                sys.stdout.write("\n")
                sys.stdout.flush()
                print(f"Upload complete: {filename}")
        except Exception as e:
            print(f"\nUpload failed: {e}")

    def upload_file_timeout(self, local_path, timeout=10):
        """
        Uploads file with a socket timeout to mitigate 'hang at 100%' issues.
        Returns: True if successful (or 100% sent), False otherwise.
        """
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), local_path)
        filename = os.path.basename(local_path)
        file_size = os.path.getsize(local_path)
        self.bytes_sent = 0

        def progress_callback(data):
            self.bytes_sent += len(data)
            percent = (self.bytes_sent / file_size) * 100
            sys.stdout.write(
                f"\rUploading {filename}: {percent:.1f}% "
                f"({self.bytes_sent}/{file_size} bytes)"
            )
            sys.stdout.flush()

        socket.setdefaulttimeout(timeout)

        try:
            with ImplicitFTP_TLS(context=self.ftp_context) as ftps:
                ftps.connect(host=self.ip, port=990)
                ftps.login(user="bblp", passwd=self.access_code)
                ftps.prot_p()
                ftps.set_pasv(True)
                
                with open(local_path, 'rb') as f:
                    print(f"Starting upload: {filename}")
                    try:
                        ftps.storbinary(f"STOR {filename}", f, callback=progress_callback)
                    except (socket.timeout, TimeoutError, ssl.SSLError):
                        if self.bytes_sent >= file_size:
                            sys.stdout.write("\n")
                            print("Note: Transfer timed out but 100% of bytes sent. Proceeding.")
                        else:
                            raise 

                sys.stdout.write("\n")
                print(f"Upload complete: {filename}")
                return True
        except Exception as e:
            sys.stdout.write("\n")
            print(f"Upload failed: {e}")
            return False
        finally:
            socket.setdefaulttimeout(None)

def load_printer_config(name=None, config_path=None):
    """
    Loads printer connection details from a YAML config file.

    :param name: Which printer entry to use. Defaults to the file's 'active_printer'.
    :param config_path: Path to the YAML file. Defaults to 'printer_config.yaml'
                        next to this script.
    :returns: dict with keys 'ip', 'access_code', 'serial'.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "printer_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Copy printer_config.example.yaml to printer_config.yaml and fill in your values."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    printers = config.get("printers", {})
    if name is None:
        name = config.get("active_printer")
    if name not in printers:
        raise KeyError(f"Printer '{name}' not found in {config_path}. "
                       f"Available: {list(printers.keys())}")

    return printers[name]


if __name__ == "__main__":
    cfg = load_printer_config()

    my_a1_mini = BambuPrinter(cfg["ip"], cfg["access_code"], cfg["serial"])

    my_a1_mini.connect()
    time.sleep(1)  # Allow MQTT connection to stabilize

    # Home all axes first to establish a reference position
    #my_a1_mini.send_gcode("G28")
    #time.sleep(10)  # Wait for homing to complete

    # Move print head to maximum X, Y and Z (180mm on the A1 Mini)
    my_a1_mini.send_gcode("G0 X180 Y180 Z180 F1200")
    print("Moving to max X, Y and Z position (180mm, 180mm, 180mm)")

    #filepath = "testPrints/cylinderFast.3mf"
    #filepath = "testPrints/bed_scraper_a1mini.gcode.3mf"
    #filepath = "testPrints/Cat_Toys_V2_-_Complete_Project_file_-_normal_speed_Multicolor with AMS.gcode.3mf"
    filepath = "testPrints/Shoe_Horn_3MF.gcode.3mf"



    remote_filename = os.path.basename(filepath)  # file lands at SD card root on upload
    my_a1_mini.enable_debug_listener()
    my_a1_mini.upload_file_timeout(filepath) # Use the timeout version or the file might stall, default is 10s use bigger numbers for bigger files
    my_a1_mini.start_print(remote_filename)
    my_a1_mini.waitUntilPrintFinished()
    print("done")

    my_a1_mini.disconnect()