# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 08:57:57 2019

@author: Aaron Shi
"""

from matplotlib import pyplot as plt
import pylab
import numpy as np
import cv2
from sklearn import datasets

faces = datasets.fetch_olivetti_faces()

for i in range(10):
    face = faces.images[i]
    plt.subplot(1, 10, i + 1)
    plt.imshow(face.reshape((64, 64)), cmap = 'gray')
    plt.axis('off')
    