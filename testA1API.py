from printerclass import BambuPrinter
import time

my_a1_mini = BambuPrinter("172.20.10.2", "14668855", "0309CA460401528")


my_a1_mini.connect()
my_a1_mini.blink_light()