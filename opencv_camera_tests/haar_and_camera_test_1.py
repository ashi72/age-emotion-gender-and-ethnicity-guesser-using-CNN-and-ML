# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:43:34 2019

@author: Aaron Shi
"""

import numpy as np
import cv2

webcam = cv2.VideoCapture(1)

face_csc = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#vid = cv2.VideoWriter('output.avi', fourcc, 6, (640,480))
while(True):
    
    #(true/false for a frame, datatype ==uint8 -> 0-255)
    tf, gray = webcam.read()
    
    #vid.write(frame)
    #print(tf)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #parameters(img, scaleFactor, minNeighbors(how accurate/fine filter), flags, min_size)
    faces = face_csc.detectMultiScale(gray, 1.4 , 4)

    #faces contains (x,y,w,h) -> coordinate for lower left of rectangle around face
    for (x,y,w,h) in faces:
        cv2.rectangle(gray, (x,y), (x+w,y+h), (0,255,0), 3)    
    
    cv2.imshow('Single Frame', gray)
    key = cv2.waitKey(1)
    if (key==27): 
        break
    

webcam.release()
#vid.release()
cv2.destroyAllWindows()