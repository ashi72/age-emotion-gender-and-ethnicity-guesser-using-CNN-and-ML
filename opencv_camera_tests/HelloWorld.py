# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 08:40:43 2019

@author: Aaron Shi
"""

import cv2

img = cv2.imread('sunset.jpg',1)

#px = img[100,100]
#print(px)

#blue = img[:,:,0]
#print (blue)

#print(img.shape)
#print(img.size)



print(img)
cv2.imshow('firstImg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
