import numpy as np
import cv2
import time
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


class USBCamera():
    def __init__(self, width=640, height=480):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, width)
        self.cap.set(4, height)

    def getFrame(self):
        rtn_val, frame = self.cap.read()
        if rtn_val:
            return frame
        else:
            print ("Failed to capture frame!")
            return None

    def isOpened(self):
        if self.cap:
            return self.cap.isOpened()
        else:
            return False

    def __del__(self):
        if self.cap:
            self.cap.release()
        print ("Cleanly exited Camera")


