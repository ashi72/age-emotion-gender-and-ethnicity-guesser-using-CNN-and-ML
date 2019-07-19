# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:51:44 2019

@author: Aaron Shi
"""

import numpy as np
import cv2

webcam = cv2.VideoCapture(0)

while(True):
    
    #(true/false for a frame, datatype ==uint8 -> 0-255)
    tf, frame = webcam.read()
    
    #print(tf)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Single Frame', gray)
    key = cv2.waitKey(1)
    if (key==27): 
        break
    

webcam.release()
cv2.destroyAllWindows()