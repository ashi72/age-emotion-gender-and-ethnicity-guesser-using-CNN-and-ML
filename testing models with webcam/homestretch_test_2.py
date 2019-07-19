# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:35:26 2019

@author: Aaron Shi
"""
import pickle
#import matplotlib
#import pylab
#from sklearn import datasets
#import json
#import IPython
#import sklearn as sk
import matplotlib.pyplot as plt
#from sklearn.svm import SVC
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score, KFold
#from scipy.stats import sem
from scipy.ndimage import zoom
import numpy as np
import cv2

loaded_model = pickle.load(open('finalized_model_3.sav', 'rb'))

webcam = cv2.VideoCapture(0)

face_csc = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
'''
while(True):
    
    tf, frame = webcam.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_csc.detectMultiScale(gray, 1.4 , 4)

    for (x,y,w,h) in faces:
        original_extracted_face = gray[y:y+h, x:x+w]
    
        #cv2.rectangle(gray, (x,y), (x+w,y+h), (0,255,0), 3)    
        extracted_face=cv2.resize(original_extracted_face, (64, 64), interpolation = cv2.INTER_AREA )
    
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(gray, 'text', (x,y), font, 4, (255,255,0), 2, cv2.LINE_AA)
    cv2.imshow('Single Frame', gray)
    key = cv2.waitKey(1)
    if (key==27): 
        break
   
  '''
while(True):
    tf, frame = webcam.read()
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_csc.detectMultiScale(gray, 1.4 , 7)
    
    for(x, y, w, h) in faces:
        
        original_extracted_face = gray[y:y+h, x:x+w]
        cv2.rectangle(gray, (x,y), (x+w,y+h), (0,255,0), 3)    
        
        extracted_face=cv2.resize(original_extracted_face, (64, 64), interpolation = cv2.INTER_AREA )
        #if (loaded_model.predict(extracted_face).reshape(1,-1))==1:
        #    text = "Smiling"
        #else:
        #    text = "not smiling"
        #text = "smiling:{0}".format(loaded_model.predict(extracted_face.reshape(1,-1))==1)
        #text = "temp"
        text ="smiling: {0}".format(loaded_model.predict(extracted_face.reshape(1, -1))==1)
        cv2.putText(gray, text , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('SingleFrame', gray)
    key = cv2.waitKey(1)
    if (key==27): 
        break
        

webcam.release()
#vid.release()
cv2.destroyAllWindows()