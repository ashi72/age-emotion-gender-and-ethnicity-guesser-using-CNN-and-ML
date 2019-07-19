# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:04:24 2019

@author: Aaron Shi
"""

import cv2

img = cv2.imread("bs_headshot_2.jpg")
#face_csc = cv2.CascadeClassifier('haarcasacade_frontalface_default.xml')

face_csc = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#parameters(img, scaleFactor, minNeighbors(how accurate/fine filter), flags, min_size)
faces = face_csc.detectMultiScale(gray, 2 , 3)

#faces contains (x,y,w,h) -> coordinate for lower left of rectangle around face
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
    
cv2.imshow('img', img)
cv2.waitKey(0) 