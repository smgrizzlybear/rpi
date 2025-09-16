from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()
picam2.start_preview(Preview.QTGL)  # Open preview window
picam2.start()
time.sleep(10)  # keep preview open for 10 sec
picam2.stop()
