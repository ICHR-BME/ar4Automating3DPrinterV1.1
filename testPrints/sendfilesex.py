from printerclass import BambuPrinter
import time

my_a1_mini = BambuPrinter("192.168.137.66", 
                          "14668855", 
                          "0309CA460401528") #dont share me
my_a1_mini.connect()
my_a1_mini.blink_light()
my_a1_mini.upload_file_timeout("tst.gcode.3mf") # Use the timeout version or the file might stall, default is 10s use bigger numbers for bigger files
my_a1_mini.start_print("tst.gcode.3mf") # Print is on sd card, now we can start