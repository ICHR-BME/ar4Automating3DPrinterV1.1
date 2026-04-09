import paho.mqtt.client as mqtt
import json
import ssl
import ftplib
import socket
import os
import sys
import time

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
        
        # SSL configuration for MQTT (Bambu uses self-signed certs)
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
        """Establishes connection to the printer's MQTT broker."""
        self.client.connect(self.ip, 8883, 60)
        self.client.loop_start()  # Starts background thread for networking
        print(f"Connected to printer {self.serial}")

    def disconnect(self):
        """Stops the background loop and disconnects MQTT."""
        self.client.loop_stop()
        self.client.disconnect()

    def enable_debug_listener(self):
        """Sets the callback and subscribes without blocking the script."""
        def on_message(client, userdata, msg):
            payload = json.loads(msg.payload.decode())
            self._parse_message(payload)
        
        self.client.on_message = on_message
        self.client.subscribe(self.topic_subscribe)
        print(f"Listener active for {self.serial}")

    def set_on_finish(self, func):
        """Pass a function here to be called when the print finishes."""
        self.on_finish_callback = func

    def _parse_message(self, data):
        """Internal parser for json """

        print_data = data.get("print", {})
        if not print_data:
            return

        # 1. Handle Percentage (Only exists in some messages)
        if "mc_percent" in print_data:
            percent = print_data.get("mc_percent")
            print(f"--> Progress: {percent}%")

        # 2. Handle Temperatures 
        if "nozzle_temper" in print_data:
            temp = round(print_data.get("nozzle_temper"), 3)
            print(f"--> Nozzle: {temp}°C")
            
        if "bed_temper" in print_data:
            bed_temp = round(print_data.get("bed_temper"), 3)
            print(f"--> Bed: {bed_temp}°C")

        # 3. Handle Printing State
        if "gcode_state" in print_data:
            state = print_data.get("gcode_state")
            print(f"--> Printer State: {state}")

        # 4. Handle Layer Number
        if "layer_num" in print_data:
            layer = print_data.get("layer_num")
            print(f"--> Current Layer: {layer}")


        if "gcode_state" in print_data:
            new_state = print_data.get("gcode_state")
            
            # Check if the state has actually changed to avoid spamming
            if new_state != self._last_state:
                print(f"--> Printer State Changed: {self._last_state} -> {new_state}")
                
                # TRIGGER: When state becomes 'FINISH'
                if new_state == "FINISH" and self.on_finish_callback:
                    print("!!! Print Finished Detected !!!")
                    self.on_finish_callback()
                
                # Update the tracker
                self._last_state = new_state


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
        """Homes all axes using G28."""
        self.send_gcode("G28")
        print("Command sent: Homing")

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

if __name__ == "__main__":
    my_a1_mini = BambuPrinter("172.20.10.2", "14668855", "0309CA460401528")
    my_a1_mini.connect()
    time.sleep(1)  # Allow MQTT connection to stabilize

    # Home all axes first to establish a reference position
    my_a1_mini.send_gcode("G28")
    time.sleep(10)  # Wait for homing to complete

    # Move print head to maximum X and Z (180mm on the A1 Mini)
    my_a1_mini.send_gcode("G0 X180 Z180 F1200")
    print("Moving to max X and Z position (180mm, 180mm)")

    time.sleep(5)
    my_a1_mini.disconnect()