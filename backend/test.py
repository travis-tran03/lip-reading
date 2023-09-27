import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2 as cv

arr = np.array([1, 2, 2, 3, 2, 3, 5, 4])

for x in range(len(arr)):
    arr[:'2'] = 7

print(arr)
'''
folderDir = 'C:/Users/Crolw/OneDrive/Documents/GitHub/lip-reading/backend/model_loss'
for image in os.listdir(folderDir):
    if (image.endswith('.png')):
        img = cv.imread(folderDir + '/' + image)
        cv.imshow('img', img)
        cv.waitKey(1000)
'''


