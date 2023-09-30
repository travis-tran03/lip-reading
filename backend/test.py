import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2 as cv
import matplotlib.pyplot as plt

path = 'C:/Users/Crolw/OneDrive/Documents/GitHub/lip-reading/backend/Histroy_TrainVal.png'

img = cv.imread(path)

greyImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

(width, height) = np.array(greyImg).shape
left = int((width-10))
right = int((height-500))
print(f'left: {left} : right: {right}')
cropImg = img[right:height, 0: left]

cv.imshow('greyimg', greyImg)
cv.imshow('cropImg', cropImg)
cv.waitKey(0)

'''
folderDir = 'C:/Users/Crolw/OneDrive/Documents/GitHub/lip-reading/backend/model_loss'
for image in os.listdir(folderDir):
    if (image.endswith('.png')):
        img = cv.imread(folderDir + '/' + image)
        cv.imshow('img', img)
        cv.waitKey(1000)
'''


