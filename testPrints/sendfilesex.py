from printerclass import BambuPrinter, load_printer_config
import time

# Credentials come from printer_config.yaml (see printer_config.example.yaml)
# instead of being hard-coded here.
cfg = load_printer_config("a1_mini_2")
my_a1_mini = BambuPrinter(cfg["ip"], cfg["access_code"], cfg["serial"])
my_a1_mini.connect()
my_a1_mini.blink_light()
my_a1_mini.upload_file_timeout("tst.gcode.3mf") # Use the timeout version or the file might stall, default is 10s use bigger numbers for bigger files
my_a1_mini.start_print("tst.gcode.3mf") # Print is on sd card, now we can start