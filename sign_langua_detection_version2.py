
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
        horizontal_axis, vertical_axis, width, height = hand ['bbox']
        
        img_white = np.ones((img_size, img_size, 3), np.uint8)* 255
        
        img_crop = img[vertical_axis-offset : vertical_axis + height + offset, horizontal_axis-offset : horizontal_axis + width + offset]
        img_crop_shape = img_crop.shape
        
        aspect_ratio = height/width
        
        if aspect_ratio > 1:
            size_of_image = img_size / height
            weight_cal = math.ceil(size_of_image*width)
            img_resize = cv2.resize(img_crop, (weight_cal, img_size))
            img_resize_shape = img_resize.shape
            
            width_gap = math.ceil ((img_size-weight_cal)/2)
            img_white[: ,width_gap: weight_cal + width_gap]
            
            
        