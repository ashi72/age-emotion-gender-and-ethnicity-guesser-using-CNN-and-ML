# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:03:45 2019

@author: Aaron Shi
"""

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
#import matplotlib.pyplot as plt
#from sklearn.svm import SVC
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score, KFold
#from scipy.stats import sem
#from scipy.ndimage import zoom
import numpy as np
import cv2


def extract_face_features(gray, detected_face, offset_coefficients):
    (x, y, w, h) = detected_face
    horizontal_offset = offset_coefficients[0] * w
    vertical_offset = offset_coefficients[1] * h
    extracted_face = gray[y+int(vertical_offset):y+h, 
                      (x+int(horizontal_offset)):(x-int(horizontal_offset)+w)]
    new_extracted_face=cv2.resize(extracted_face, (64, 64), interpolation = cv2.INTER_AREA )        
    return new_extracted_face

def detect_face(frame):
    face_csc = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    #frame = frame.astype('uint8')
    frame = np.asarray(frame, dtype=np.uint8)    
    #frame = np.array(frame).astype(np.int32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    detected_faces = face_csc.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    return gray, detected_faces


loaded_model = pickle.load(open('finalized_model_3.sav', 'rb'))

webcam = cv2.VideoCapture(0)

face_csc = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')

while(True):
    tf, frame = webcam.read()
    #frame = frame.astype('uint8')
    #frame = np.asarray(frame, dtype=np.uint8)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray, detected_faces = detect_face(frame)
    
    #faces = face_csc.detectMultiScale(gray, 1.4 , 7)
    
    face_index = 0
    for(x, y, w, h) in detected_faces:
        
        #original_extracted_face = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)    
        
        extracted_face=extract_face_features(gray, (x, y, w, h), (0.2, 0.15))
        
        prediction = loaded_model.predict(extracted_face.reshape(1, -1))==1
        
        #frame[face_index * 64: (face_index+1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_BGR2GRAY)    cv2.COLOR_BAYER_BG2BGR)#  
        frame[face_index * 64: (face_index + 1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)

        if prediction:
            cv2.putText(frame, "SMILING",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
        else:
            cv2.putText(frame, "not smiling",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)

        face_index += 1
    cv2.imshow('SingleFrame', frame)
    key = cv2.waitKey(1)
    if (key==27): 
        break
        

webcam.release()
#vid.release()
cv2.destroyAllWindows()