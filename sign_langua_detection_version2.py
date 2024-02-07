
# ========= IMPORTS ==========
import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import math
import numpy

# ========= VIDEO CAPTURE ========
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
img_size = 300
counter = 0

# ========= ACTUAL CODES ==========
