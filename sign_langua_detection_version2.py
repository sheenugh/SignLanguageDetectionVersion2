
# ========= IMPORTS ==========
import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import math
import numpy as np
import mediapipe as mp


# ========= VIDEO CAPTURE ========
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
img_size = 200
counter = 0

folder = "Data\Letters\Letter Q "   

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
            width_cal = math.ceil(size_of_image*width)
            img_resize = cv2.resize(img_crop, (width_cal, img_size))
            img_resize_shape = img_resize.shape
            
            width_gap = math.ceil ((img_size-width_cal)/2)
            img_white[: ,width_gap: width_cal + width_gap] = img_resize
            
        else:
            size_of_image = img_size / width
            height_cal = math.ceil(size_of_image*height)
            img_resize = cv2.resize(img_crop, (img_size, height_cal))
            img_resize_shape = img_resize.shape
            
            height_gap = math.ceil ((img_size-height_cal)/2)
            img_white[height_gap : height_cal + height_gap, : ] = img_resize
        
        cv2.imshow('ImageCrop', img_crop)
        cv2.imshow('ImageWhite', img_white)
        
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s') :
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', img_white)
        print(counter)
        
        
            
            
            
        