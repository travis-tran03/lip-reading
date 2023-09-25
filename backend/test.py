import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2 as cv

folderDir = 'C:/Users/Crolw/OneDrive/Documents/GitHub/lip-reading/backend/model_loss'
for image in os.listdir(folderDir):
    if (image.endswith('.png')):
        img = cv.imread(folderDir + '/' + image)
        cv.imshow('img', img)
        cv.waitKey(1000)



