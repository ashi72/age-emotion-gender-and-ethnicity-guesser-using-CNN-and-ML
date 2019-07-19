# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:55:26 2019

@author: Aaron Shi
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

cam = cv2.VideoCapture(1)

#tf, frame = cam.read()
#print(tf)

#cv2.imshow('SingleFrame', frame)
#cv2.waitKey(0)
while(True):
    tf, frame = cam.read()
    #print tf
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('SingleFrame', gray)
    key = cv2.waitKey(1)
    if key ==27:
        break
   
cam.release()
cv2.destroyAllWindows()
    