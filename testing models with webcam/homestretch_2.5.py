# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:14:07 2019

@author: Aaron Shi
"""

import pickle
import matplotlib
import pylab
from sklearn import datasets
import json
import IPython
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem
from scipy.ndimage import zoom
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')


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

def make_map(facefile):
    c1_range = np.linspace(0, 0.35)
    c2_range = np.linspace(0, 0.3)
    result_matrix = np.nan * np.zeros_like(c1_range * c2_range[:, np.newaxis])
    gray, detected_faces = detect_face(cv2.imread(facefile))
    for face in detected_faces[:1]:
        for ind1, c1 in enumerate(c1_range):
            for ind2, c2 in enumerate(c2_range):
                extracted_face = extract_face_features(gray, face, (c1, c2))
                result_matrix[ind1, ind2] = (loaded_model.predict(extracted_face.reshape(1, -1))==1)
    return (c1_range, c2_range, result_matrix)

loaded_model = pickle.load(open('finalized_model_3.sav', 'rb'))

r1 = make_map("SMILE_2.jpg")
r2 = make_map("SAD_2.jpg")

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.title('not smiling image')
plt.pcolormesh(r1[0], r1[1], r1[2])
plt.colorbar()
plt.xlabel('horizontal stretch factor c1')
plt.ylabel('vertical stretch factor c2')

plt.subplot(132)
plt.title('smiling image')
plt.pcolormesh(r2[0], r2[1], r2[2])
plt.colorbar()
plt.xlabel('horizontal stretch factor c1')
plt.ylabel('vertical stretch factor c2')

plt.subplot(133)
plt.title('correct settings for both images simultaneously')
plt.pcolormesh(r1[0], r1[1], (r1[2]==1) & (r2[2]==0), cmap='autumn')
plt.colorbar()
plt.xlabel('horizontal stretch factor c1')
plt.ylabel('vertical stretch factor c2')

plt.tight_layout()
