# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 09:33:23 2019

@author: Aaron Shi
"""

import numpy as np
import cv2
import tensorflow.keras
import pickle
import matplotlib
import sklearn as sk
import matplotlib.pyplot as plt

filepath = "C:/Users/Aaron Shi/dev/python/bluestamp/facial_1"
cnn = tensorflow.keras.models.load_model(filepath)

emo = ['Angry', 'Anger', 'Surprise', 'Happy',
           'Sad', 'Surprise', 'Neutral']

#emo = ['Angry', 'Disgust', 'Fear', 'Happy',
#           'Sad', 'Surprise', 'Neutral']

#emo = ['other', 'other', 'other', 'Happy', 'Sad', 'other', 'Neutral']

def display_text(prediction):
    maxx = 0
    text = ""
    for i in range(0,7):
        if prediction[0][i] > maxx:
            text = emo[i]            
            maxx = prediction[0][i]
    return text


def prepare(img_array, IMG_SIZE):
    #IMG_SIZE = 48
    #img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    #print(img_array)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def extract_face_features(gray, detected_face, offset_coefficients):
    (x, y, w, h) = detected_face
    horizontal_offset = offset_coefficients[0] * w
    vertical_offset = offset_coefficients[1] * h
    extracted_face = gray[y+int(vertical_offset)-int(0.10*h):(y+h),#-int(0.10*h), 
                      (x+int(horizontal_offset)):(x-int(horizontal_offset)+w)]
    new_extracted_face=cv2.resize(extracted_face, (48, 48), interpolation = cv2.INTER_AREA )        
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

#def multi_class_text(extracted_face_shape)
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
        
        prep = prepare(extracted_face, 48)
        
        prediction = cnn.predict(prep)
        
        #print(prediction)
        #prediction = loaded_model.predict(extracted_face.reshape(1, -1))==1
        
        #frame[face_index * 64: (face_index+1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_BGR2GRAY)    cv2.COLOR_BAYER_BG2BGR)#  
        #frame[face_index * 64: (face_index + 1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)
        
        #print(extracted_face.shape)
        #print(extracted_face.shape.reshape(1, -1).shape)
        #predicted = cnn.predict(extracted_face.reshape(1, -1))
        #print(predicted)
        #frame[face_index * 48: (face_index + 1) * 48, -49:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)
        
        text = display_text(prediction)
        #if prediction:
        cv2.putText(frame, text,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
        #else:
        #    cv2.putText(frame, "not smiling",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)

        face_index += 1
    cv2.imshow('SingleFrame', frame)
    key = cv2.waitKey(1)
    if (key==27): 
        break
        

webcam.release()
#vid.release()
cv2.destroyAllWindows()