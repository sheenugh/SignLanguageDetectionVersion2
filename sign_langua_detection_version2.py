
# ========= IMPORTS ==========
import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import math
import numpy as np

# ========= VIDEO CAPTURE ========
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
img_size = 300
counter = 0

folder = "C:\Users\Myline\Desktop\[PLD] Final Project\SignLanguageDetectionVersion2\Data"

# ========= ACTUAL CODES ==========
# - Calling for data collection
while True :
    success, img = cap.read ()
    hands , img = detector.findHands(img)
    if hands:
        hand = hands [0]
        x,y,w,h = hand ['bbox']
        
        img_white = np.ones((img_size, img_size, 3), np.uint8)* 255
        
        img_crop = img[y-offset : y + h + offset, x-offset : x + w + offset]
        img_crop_shape = img_crop.shape
        