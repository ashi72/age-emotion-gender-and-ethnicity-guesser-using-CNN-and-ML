# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 08:34:42 2019

@author: Aaron Shi
"""

#import matplotlib
#import pylab
from sklearn import datasets
import json
#import IPython
#import sklearn as sk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import cross_val_score, KFold
#from scipy.stats import sem
import pickle

def print_faces(images, tr, top_n):
    # set up the figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
        p.imshow(faces.images[i], cmap=plt.cm.bone)
        
        # label the image with the target value
        if tr[str(i)] == False:
            p.text(0, 14, "notsmiling")
            p.text(0, 60, str(i))
        else:
            p.text(0, 14, "smiling")
            p.text(0, 60, str(i))
                        
faces = datasets.fetch_olivetti_faces()

with open("results_3.xml","r") as f:
    d = json.load(f)
    
#print_faces(faces, d, 400)

svc_1 = SVC(kernel='linear')

target = [d[i] for i in d]
target = np.array(target).astype(np.int32)

x,y = faces.data[:-1], target[:-1]


#svc_1.fit(faces.data, target)
svc_1.fit(x,y)
#index = randint(0,400)

print('this person is smiling:{0}'.format(svc_1.predict(faces.data[-1, :].reshape(1,-1))==1))

plt.imshow(faces.images[-1],cmap = plt.cm.gray, interpolation = "nearest")

plt.show()

filename = 'finalized_smile_take2.sav'
pickle.dump(svc_1, open(filename, 'wb'))

