# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:15:27 2019

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

a_cnn = tensorflow.keras.models.load_model("C:/Users/Aaron Shi/dev/python/bluestamp/age_2")
g_cnn = tensorflow.keras.models.load_model("C:/Users/Aaron Shi/dev/python/bluestamp/app_gender_2")
r_cnn = tensorflow.keras.models.load_model("C:/Users/Aaron Shi/dev/python/bluestamp/race_1")

emo = ['Angry', 'Sad', 'Surprise', 'Happy',
           'Sad', 'Surprise', 'Neutral']

age_classes = ["infant", "child", "child", "adolescent", "young-adult", "adult", "middle-aged", "aged", "elderly"]

gen = ["male", "female"]

race = ["white", "black", "asian", "indian", ""]
#emo = ['Angry', 'Disgust', 'Fear', 'Happy',
#           'Sad', 'Surprise', 'Neutral']

#emo = ['other', 'other', 'other', 'Happy', 'Sad', 'other', 'Neutral']

def display_text(prediction, arr):
    maxx = 0
    text = ""
    for i in range(0,len(arr)):
        if prediction[0][i] > maxx:
            text = arr[i]            
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
            minNeighbors=10,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    return gray, detected_faces

#def multi_class_text(extracted_face_shape)
loaded_model = pickle.load(open('finalized_model_3.sav', 'rb'))

webcam = cv2.VideoCapture(1)

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
        prediction1 = cnn.predict(prep)    
        text = display_text(prediction1, emo)

        prep2 = prepare(extracted_face, 64)
        pred_a = a_cnn.predict(prep2)
        pred_g = g_cnn.predict(prep2)
        pred_r = r_cnn.predict(prep2)
        
        a = display_text(pred_a, age_classes)        
        g = display_text(pred_g, gen)        
        r = display_text(pred_r, race)
        
        #a = ""
        #scale = 1
        fontScale = 0.85    #    min(w,h)/(25/scale)
        cv2.putText(frame, text+" "+r+" "+g+" "+a,(x,y), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0,0,255), 2)
        
        face_index += 1
    cv2.imshow('SingleFrame', frame)
    key = cv2.waitKey(1)
    if (key==27): 
        break
        

webcam.release()
#vid.release()
cv2.destroyAllWindows()