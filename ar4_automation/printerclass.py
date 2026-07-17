import paho.mqtt.client as mqtt
import json
import ssl
import ftplib
import socket
import struct
import fcntl
import ipaddress
import threading
import concurrent.futures
import os
import re
import sys
import time
import hashlib
import zipfile
import yaml

# Relative file/config paths are resolved against the repo root (the parent of
# this package), so callers can pass e.g. "gcode/foo.3mf" from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _local_ipv4_interfaces():
    """Return [(name, ip, netmask)] for every non-loopback IPv4 interface."""
    results = []
    for _idx, name in socket.if_nameindex():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            ifreq = struct.pack("256s", name.encode()[:15])
            ip = socket.inet_ntoa(
                fcntl.ioctl(s.fileno(), 0x8915, ifreq)[20:24])   # SIOCGIFADDR
            mask = socket.inet_ntoa(
                fcntl.ioctl(s.fileno(), 0x891B, ifreq)[20:24])   # SIOCGIFNETMASK
        except OSError:
            continue  # interface has no IPv4 address (down / unconfigured)
        finally:
            s.close()
        if not ip.startswith("127."):
            results.append((name, ip, mask))
    return results

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
        """ip: printer's LAN address; access_code: 8 digits from the printer
        settings screen; serial: printer serial number."""
        self.ip = ip
        self.access_code = access_code
        self.serial = serial
        self.topic_publish = f"device/{self.serial}/request"
        self.topic_subscribe = f"device/{self.serial}/report"

        self.on_finish_callback = None
        self._last_state = None

        # Live status panel: latest known values, redrawn in place on one line.
        self.debug = False          # set True to dump raw MQTT payloads
        self.status = {}            # e.g. {"state": ..., "progress": ..., "nozzle": ...}

        self._print_finished = False

        # watchdog rebuilds the connection (rediscovering the IP if needed)
        # whenever the link goes quiet, e.g. after a hotspot drop
        self.stale_timeout = 30.0        # s without any message -> assume dead
        self._watchdog_poll = 5.0        # s between watchdog checks
        self._connected = threading.Event()
        self._connack_event = threading.Event()
        self._connack_rc = None
        self._last_msg_time = time.time()
        self._listener_enabled = False
        self._nudged = False
        self._watchdog = None
        self._watchdog_stop = threading.Event()

        # Setup the MQTT Client (SSL because Bambu uses self-signed certs)
        self._setup_client()

        # Setup FTP SSL Context for file transfers
        self.ftp_context = ssl.create_default_context()
        self.ftp_context.check_hostname = False
        self.ftp_context.verify_mode = ssl.CERT_NONE
        self.ftp_context.options |= ssl.OP_IGNORE_UNEXPECTED_EOF

    # ------------------------------------------
    # MQTT Communication Methods
    # ------------------------------------------

    def _setup_client(self):
        """Fresh MQTT client. Also called on watchdog rebuilds, so callbacks
        must be methods, not one-shot closures."""
        self.client = mqtt.Client()
        self.client.username_pw_set("bblp", self.access_code)
        self.client.tls_set(cert_reqs=ssl.CERT_NONE)
        self.client.tls_insecure_set(True)
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def _on_connect(self, client, _userdata, _flags, rc):
        self._connack_rc = rc
        self._connack_event.set()
        if rc != 0:
            return
        self._connected.set()
        self._last_msg_time = time.time()
        # resubscribe every connect (broker drops the session on disconnect);
        # the pushall catches up on anything missed while offline
        client.subscribe(self.topic_subscribe)
        client.publish(self.topic_publish, json.dumps(
            {"pushing": {"sequence_id": "0", "command": "pushall"}}))

    def _on_disconnect(self, _client, _userdata, rc):
        self._connected.clear()
        if rc != 0:
            print(f"\nLost connection to printer {self.serial} (rc={rc}); "
                  f"reconnecting...")

    def _on_message(self, _client, _userdata, msg):
        self._last_msg_time = time.time()
        if not self._listener_enabled:
            return
        payload = json.loads(msg.payload.decode())
        # Debug: dump command acks (e.g. the response to project_file)
        if self.debug and "print" in payload and payload["print"].get("command") in ("project_file", "push_status"):
            print("\nRAW:", json.dumps(payload, indent=2))
        self._parse_message(payload)

    def connect(self):
        """Connect to the printer's MQTT broker. Falls back to network
        discovery if the configured IP is stale, and starts the reconnect
        watchdog."""
        self._establish()
        print(f"Connected to printer {self.serial}")
        self._start_watchdog()

    def _establish(self, connack_timeout=10.0):
        """Discover if needed, connect, wait for CONNACK. The wait matters:
        publishes before the session is accepted vanish silently, and a bad
        access code otherwise looks like a successful connection."""
        if not self._is_reachable(self.ip):
            print(f"Printer not reachable at {self.ip}; searching the local network...")
            found_ip = self.discover_ip()
            if not found_ip:
                found_ip = self.discover_ip_by_scan()
            if found_ip:
                print(f"Found printer {self.serial} at {found_ip} (was {self.ip}).")
                self.ip = found_ip
            else:
                print(f"Warning: printer {self.serial} not found on the network; "
                      f"trying {self.ip} anyway.")

        self._connack_event.clear()
        self._connack_rc = None
        self.client.connect(self.ip, 8883, 60)
        self.client.loop_start()  # Starts background thread for networking
        if not self._connack_event.wait(connack_timeout):
            raise ConnectionError(
                f"Printer {self.serial} at {self.ip}: no MQTT CONNACK within "
                f"{connack_timeout:.0f}s.")
        if self._connack_rc != 0:
            self.client.loop_stop()
            raise ConnectionError(
                f"Printer {self.serial} at {self.ip} rejected the MQTT login "
                f"(rc={self._connack_rc}). Usually a wrong access_code in "
                f"printer_config.yaml — check the code on the printer's screen "
                f"(Settings → WLAN)."
            )

    def _start_watchdog(self):
        if self._watchdog is not None and self._watchdog.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog = threading.Thread(
            target=self._connection_watchdog, daemon=True,
            name=f"bambu-watchdog-{self.serial}")
        self._watchdog.start()

    def _connection_watchdog(self):
        """After stale_timeout of silence, nudge with a pushall (may just be
        idle); if still quiet, rebuild the client and reconnect. Retries every
        poll until the printer is back."""
        while not self._watchdog_stop.wait(self._watchdog_poll):
            silent = time.time() - self._last_msg_time
            if silent < self.stale_timeout:
                self._nudged = False
                continue
            if self._connected.is_set() and not self._nudged:
                self._nudged = True
                self.client.publish(self.topic_publish, json.dumps(
                    {"pushing": {"sequence_id": "0", "command": "pushall"}}))
                continue
            print(f"\nNo messages from printer {self.serial} for {silent:.0f}s; "
                  f"rebuilding the connection...")
            if self._reconnect():
                self._nudged = False

    def _reconnect(self):
        """Tear down and reconnect from scratch. Safe to call repeatedly."""
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass
        self._connected.clear()
        self._setup_client()
        try:
            self._establish()
        except (ConnectionError, OSError, ssl.SSLError) as e:
            print(f"Reconnect to printer {self.serial} failed ({e}); "
                  f"will retry in {self._watchdog_poll:.0f}s.")
            return False
        print(f"Reconnected to printer {self.serial}; resuming status updates.")
        return True

    def disconnect(self):
        """Stop the watchdog and MQTT loop."""
        self._watchdog_stop.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=1.0)
            self._watchdog = None
        self._connected.clear()
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
        """Find the printer's IP from Bambu's SSDP broadcasts (multicast to
        239.255.255.250, UDP 2021/1990, serial in the USN header). None if
        not seen within timeout."""
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

        # join on every interface: 0.0.0.0 only covers the default route and
        # misses a printer on a locally hosted hotspot
        for _name, if_ip, _mask in _local_ipv4_interfaces():
            mreq = struct.pack("4s4s", socket.inet_aton(mcast_grp),
                               socket.inet_aton(if_ip))
            try:
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            except OSError:
                pass
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

    def discover_ip_by_scan(self, port=8883, timeout=0.4, verify_timeout=5.0):
        """SSDP fallback: TCP-scan private /24-or-smaller local subnets for
        the MQTT port and verify each hit is this printer. Never sweeps the
        upstream wifi network."""
        candidates = []
        for name, if_ip, mask in _local_ipv4_interfaces():
            net = ipaddress.IPv4Network(f"{if_ip}/{mask}", strict=False)
            if not net.is_private or net.prefixlen < 24:
                continue
            print(f"Scanning {net} on {name} for the printer...")
            hosts = [str(h) for h in net.hosts() if str(h) != if_ip]
            with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
                reachable = ex.map(
                    lambda h: self._is_reachable(h, port, timeout), hosts)
                candidates += [h for h, ok in zip(hosts, reachable) if ok]

        for ip in candidates:
            if self._verify_printer(ip, timeout=verify_timeout):
                return ip
            print(f"Host {ip} answers on the MQTT port but is not printer {self.serial}.")
        return None

    def _verify_printer(self, ip, timeout=5.0):
        """True if the printer at ip answers a status push on this serial's
        topic (a different printer rejects the code or never publishes)."""
        got_report = threading.Event()
        client = mqtt.Client()
        client.username_pw_set("bblp", self.access_code)
        client.tls_set(cert_reqs=ssl.CERT_NONE)
        client.tls_insecure_set(True)

        def on_connect(cl, _userdata, _flags, rc):
            if rc == 0:
                cl.subscribe(self.topic_subscribe)
                cl.publish(self.topic_publish, json.dumps(
                    {"pushing": {"sequence_id": "0", "command": "pushall"}}))

        client.on_connect = on_connect
        client.on_message = lambda *_args: got_report.set()
        try:
            client.connect(ip, 8883, 60)
        except (OSError, ssl.SSLError):
            return False
        client.loop_start()
        ok = got_report.wait(timeout)
        client.loop_stop()
        client.disconnect()
        return ok

    def enable_debug_listener(self):
        """Turn on report parsing (status panel + finish detection)."""
        self._listener_enabled = True
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

    def _send_command(self, command_dict, reconnect_wait=120.0):
        """Publish a JSON command. If the link is down, wait for the watchdog
        to restore it first (QoS-0 publishes on a dead connection vanish)."""
        if not self._connected.is_set():
            print(f"MQTT to printer {self.serial} is down; waiting up to "
                  f"{reconnect_wait:.0f}s for it to reconnect before sending...")
            if not self._connected.wait(reconnect_wait):
                print("Warning: still not connected; attempting to send anyway.")
        payload = json.dumps(command_dict)
        info = self.client.publish(self.topic_publish, payload)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"Warning: MQTT publish failed (rc={info.rc}); "
                  f"command not delivered: {payload[:120]}")
        return info.rc == mqtt.MQTT_ERR_SUCCESS

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
        gcode_filename = os.path.join(_REPO_ROOT, gcode_filename)
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
        """Upload to the SD card via FTPS. Can hang at 100% on some firmware,
        prefer upload_file_timeout."""
        local_path = os.path.join(_REPO_ROOT, local_path)
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
        """upload_file with a socket timeout to work around the hang-at-100%
        issue. True if all bytes were sent."""
        local_path = os.path.join(_REPO_ROOT, local_path)
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

# ==========================================
# G-code post-processing
# ==========================================

# markers delimiting the machine-start block in Bambu-sliced gcode
_STARTUP_BEGIN_MARKER = ";===== machine:"          # first line of machine_start_gcode
_STARTUP_END_MARKER   = ";===== nozzle load line"  # stock profile: blob squirt kept as-is
# custom profiles have no nozzle-load section; their block ends here and a
# blob squirt is synthesized instead
_HANDOFF_END_MARKER   = "; ===== hand-off to slicer first move"
_STRIP_SENTINEL       = ";===== startup stripped"


def _strip_startup_from_text(gcode):
    """Cut the machine startup block (purge, wipe, bed level, re-home) and
    replace it with a minimal heat-and-home preamble. The blob squirt and the
    print itself are kept; custom profiles without a nozzle-load section get
    a synthesized blob using temperatures salvaged from the removed block."""
    if re.search(r"^" + re.escape(_STRIP_SENTINEL), gcode, re.MULTILINE):
        return gcode  # already stripped

    # anchor at line starts: a config comment embeds a one-line copy of the
    # whole block that a plain find() would hit instead
    begin_m = re.search(r"^" + re.escape(_STARTUP_BEGIN_MARKER), gcode, re.MULTILINE)
    end_m = re.search(r"^" + re.escape(_STARTUP_END_MARKER), gcode, re.MULTILINE)
    synthesize_blob = False
    if not end_m:
        # custom profile: cut the whole block, squirt the blob ourselves
        end_m = re.search(r"^" + re.escape(_HANDOFF_END_MARKER), gcode, re.MULTILINE)
        synthesize_blob = True
    if not begin_m or not end_m or end_m.start() < begin_m.start():
        raise ValueError(
            "Startup markers not found — not a recognized Bambu machine-start "
            f"gcode block (looked for '{_STARTUP_BEGIN_MARKER}' followed by "
            f"'{_STARTUP_END_MARKER}' or '{_HANDOFF_END_MARKER}')."
        )
    begin, end = begin_m.start(), end_m.start()

    removed = gcode[begin:end]

    # salvage the temperatures the removed block would have set
    bed_heat = re.search(r"^M140 S(\d+)", removed, re.MULTILINE)
    bed_wait = re.search(r"^M190 S(\d+)", removed, re.MULTILINE)
    bed_temp = (bed_wait or bed_heat).group(1) if (bed_wait or bed_heat) else None
    nozzle_temps = [int(v) for v in re.findall(r"^M10[49] S(\d+)", removed, re.MULTILINE)]
    nozzle_temp = max(nozzle_temps) if nozzle_temps else None
    filament_type = re.search(r"^M1002 set_filament_type:.*$", removed, re.MULTILINE)

    preamble = [
        f"{_STRIP_SENTINEL} (see strip_startup_gcode in printerclass.py) =====",
        "M1002 gcode_claim_action : 2",
    ]
    if filament_type:
        preamble.append(filament_type.group(0))
    preamble.append("G392 S0 ;turn off clog detect")
    preamble.append("M104 S140 ; preheat nozzle, kept below the Z-probe limit")
    if bed_temp:
        preamble.append(f"M140 S{bed_temp} ; start heating bed")
    preamble += [
        "G90",
        "M83",
        "M220 S100 ;Reset Feedrate",
        "M221 S100 ;Reset Flowrate",
        "M1002 gcode_claim_action : 13",
        "G28 T145 ; home all axes (waits for nozzle below 145C before Z probe)",
        "M975 S1 ; turn on vibration suppression",
        "M1002 gcode_claim_action:54",
    ]
    if bed_temp:
        preamble.append(f"M190 S{bed_temp} ; wait for bed temperature")
    preamble.append("G29.2 S1 ; enable the saved bed mesh compensation")

    if not synthesize_blob:
        # the kept nozzle-load section does the blob squirt
        preamble += [
            "G1 Z5 F3000",
            "G1 X10 Y10 F20000",
        ]
    else:
        if nozzle_temp is None:
            raise ValueError(
                "No nozzle temperature (M104/M109) found in the removed "
                "startup block — cannot synthesize the nozzle-load blob.")
        preamble += [
            "; synthesized nozzle load: squirt a blob into the waste chute",
            "T1000",
            "M211 X0 Y0 Z0 ; disable soft endstops (this profile prints with them off)",
            "G1 Z10 F3000",
            "G1 X-13.5 Y0 F10000 ; park over the waste chute",
            f"M109 S{nozzle_temp} ; wait for print temperature",
            "M83",
            "G1 E20 F300 ; extrude the blob",
            "G1 E-0.5 F1800",
            "M400",
            "M106 P1 S255 ; cool the blob so it detaches",
            "M400 S2",
            "G1 X0 F18000 ; shake the blob off",
            "G1 X-13.5 F3000",
            "G1 X0 F18000",
            "G1 X-13.5 F3000",
            "M106 P1 S0",
            "M400",
        ]
    preamble += [
        f"{_STRIP_SENTINEL} end =====",
        "",
    ]
    return gcode[:begin] + "\n".join(preamble) + gcode[end:]


def strip_startup_gcode(local_path, output_path=None):
    """Strip the startup procedures from a print file; they shove the plate
    around and break the scrape automation. Takes .gcode or .gcode.3mf (every
    Metadata/*.gcode entry stripped, .md5 sidecars recomputed). Writes a
    '_stripped' copy next to the source and returns its path."""
    local_path = os.path.join(_REPO_ROOT, local_path)

    if output_path is None:
        for ext in (".gcode.3mf", ".3mf", ".gcode"):
            if local_path.endswith(ext):
                output_path = local_path[:-len(ext)] + "_stripped" + ext
                break
        else:
            raise ValueError(f"Unrecognized print file extension: {local_path}")
    else:
        output_path = os.path.join(base_dir, output_path)

    if local_path.endswith(".gcode"):
        with open(local_path, "r") as f:
            stripped = _strip_startup_from_text(f.read())
        with open(output_path, "w") as f:
            f.write(stripped)
        return output_path

    # .3mf: rewrite the gcode entries, pass everything else through
    with zipfile.ZipFile(local_path, "r") as src:
        entries = {info.filename: src.read(info.filename) for info in src.infolist()}

    gcode_names = [n for n in entries
                   if n.startswith("Metadata/") and n.endswith(".gcode")]
    if not gcode_names:
        raise ValueError(f"No Metadata/*.gcode entry found in {local_path}")

    for name in gcode_names:
        stripped = _strip_startup_from_text(entries[name].decode("utf-8"))
        entries[name] = stripped.encode("utf-8")
        md5_name = name + ".md5"
        if md5_name in entries:
            entries[md5_name] = hashlib.md5(entries[name]).hexdigest().upper().encode()

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as dst:
        for name, data in entries.items():
            dst.writestr(name, data)

    return output_path


def load_printer_config(name=None, config_path=None):
    """Load a printer's ip/access_code/serial from printer_config.yaml.
    name defaults to the file's 'active_printer'."""
    if config_path is None:
        config_path = os.path.join(_REPO_ROOT, "printer_config.yaml")

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

    #filepath = "gcode/cylinderFast.3mf"
    #filepath = "gcode/bed_scraper_a1mini.gcode.3mf"
    #filepath = "gcode/Cat_Toys_V2_-_Complete_Project_file_-_normal_speed_Multicolor with AMS.gcode.3mf"
    filepath = "gcode/Shoe_Horn_3MF.gcode.3mf"



    remote_filename = os.path.basename(filepath)  # file lands at SD card root on upload
    my_a1_mini.enable_debug_listener()
    my_a1_mini.upload_file_timeout(filepath) # Use the timeout version or the file might stall, default is 10s use bigger numbers for bigger files
    my_a1_mini.start_print(remote_filename)
    my_a1_mini.waitUntilPrintFinished()
    print("done")

    my_a1_mini.disconnect()