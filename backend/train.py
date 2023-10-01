import os

import numpy as np

from keras.callbacks import Callback, EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Adam

import cv2 as cv

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import pandas as pd

from network.architecture import neuralNetwork
from allData.resizing import crop
import allData.data as dt

phrases = ["stop navigation", "excuse me", "i am sorry", "thank you", "good bye", "i love this game", "nice to meet you", "you are welcome", "how are you", "have a good time"]
words = np.array(["begin", "choose", "connection", "navigation", "next", "previous", "start", "stop", "hello", "wed"])

SEQUENCE_LENTH = 10

def data():
    basePath = 'C:/Users/Crolw/OneDrive/Documents/GitHub/lip-reading/MIRACL-VC1_all_in_one/'

    imgs = []
    labels = []

    notCropped = []

    testData = []
    testLabel = []

    valData = []
    valLabel = []

    for i in range(1, 11): # Person
        if (i==3):
            continue
        for x in range(1, 11): # Phrase Number
            for y in range(1, 11): # Repeat Number
                path = os.path.join(basePath, f"F{str(i).zfill(2)}/phrases/{str(x).zfill(2)}/{str(y).zfill(2)}") # Full FIle Path

                newImgs, newNotCropped = dt.load_images(path, [], [])
                
                frames = []

                skipFrames = max(int(len(newImgs) / SEQUENCE_LENTH), 1)

                for frameCount in range(SEQUENCE_LENTH):
                    index = frameCount*skipFrames

                    if index > len(newImgs)-1:
                        break


                    frame = newImgs[index]

                    frames.append(frame)

                if len(frames) == SEQUENCE_LENTH:
                    if (x == 9):
                        valData.append(frames)
                        valLabel.append(phrases[x-1])
                    elif (x == 10):
                        testData.append(frames)
                        testLabel.append(phrases[x-1])
                    else:
                        imgs.append(frames)
                        labels.append(phrases[x-1])
                
                for k in range(len(newNotCropped)):
                    notCropped.append(newNotCropped[k])

    return imgs, labels, notCropped, testData, testLabel, valData, valLabel

def getData():
    basePath = 'C:/Users/Crolw/OneDrive/Documents/GitHub/lip-reading/MIRACL-VC1_all_in_one/'

    imgs = []
    labels = []

    notCropped = []

    testData = []
    testLabel = []

    valData = []
    valLabel = []

    for i in range(1, 11):
        if (i == 3):
            continue
        for x in range(1, 11):
            for y in range(1, 11):
                path = os.path.join(basePath, f"F{str(i).zfill(2)}/phrases/{str(x).zfill(2)}/{str(y).zfill(2)}")

                newImgs, newNotCropped = dt.load_images(path, [], [])



                if (x > 6 and x < 9):
                    for j in range(len(newImgs)):
                        valData.append(newImgs[j])
                        valLabel.append(phrases[x-1])
                elif (x > 8 and x < 11):
                    for j in range(len(newImgs)):
                        testData.append(newImgs[j])
                        testLabel.append(phrases[x-1])
                else:
                    for j in range(len(newImgs)):
                        imgs.append(newImgs[j])
                        labels.append(phrases[x-1])
                
                for k in range(len(newNotCropped)):
                    notCropped.append(newNotCropped[k])
    
    return imgs, labels, notCropped, testData, testLabel, valData, valLabel

def oneHotEncode(type, labels):
    tempPhrases = list(type)
    unique = np.unique(tempPhrases)

    mapping = {}
    for x in range(len(unique)):
        mapping[unique[x]] = x

    for x in range(len(tempPhrases)):
        tempPhrases[x] = mapping[tempPhrases[x]]

    onehot = to_categorical(tempPhrases, num_classes=10)

    onehotArr = []

    labels = labels.reshape(len(labels), 1)

    tempList = labels.tolist()
    for i in range(len(type)):
        for x in np.where(labels == type[i])[0]:
            tempList[x] = onehot[i]

            
    onehotArr = np.array(tempList)

    onehot = onehot.reshape(len(onehot), 10)

    onehotArr = onehotArr.reshape(len(onehotArr), 10)

    return onehotArr

def convertToNumpy(arr):
    arrList = [np.array(item) for item in arr]

    finalList = np.stack(arrList)  

    return finalList

features = []
labels = []

# Index 1 = # of Videos, Index 2: Frames (10), Index 3: Frame Number / Frame Img (91, 91)

trainImages, trainLabels, notCropped, testImages, testLabels, validationImages, validationLabels = data()

trainImages = convertToNumpy(trainImages)

cv.imshow('img', trainImages[12][4])
cv.waitKey(0)

trainLabels = convertToNumpy(trainLabels)

testImages = convertToNumpy(testImages)

testLabels = convertToNumpy(testLabels)

validationImages = convertToNumpy(validationImages)

validationLabels = convertToNumpy(validationLabels)

print(trainImages.shape)

model = neuralNetwork(trainImages.shape[1], trainImages.shape[2], trainImages.shape[3])

optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

onehotTrain = oneHotEncode(phrases, trainLabels)
onehotTest = oneHotEncode(phrases, testLabels)
onehotValidation = oneHotEncode(phrases, validationLabels)

'''
trainImages = trainImages.reshape(trainImages.shape[0], 13, trainImages.shape[1], trainImages.shape[1], 1)
validationImages = validationImages.reshape(validationImages.shape[0], 13, validationImages.shape[1], validationImages.shape[1], 1)
testImages = testImages.reshape(testImages.shape[0], 13, testImages.shape[1], testImages.shape[1], 1)
'''

earlyStoppiingCallback = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)


history = model.fit(trainImages, onehotTrain, epochs=50, batch_size=8,
                    callbacks=[earlyStoppiingCallback],
                    validation_data=(validationImages, onehotValidation),
                    validation_batch_size=8, shuffle='batch_size')

print(f'Loss: {history.history["loss"]}')
print(f'Val_Loss: {history.history["val_loss"]}')
print(f'Accuracy: {history.history["accuracy"]}')
print(f'Val_Accuracy: {history.history["val_accuracy"]}')

x = history.epoch

plt.plot(x, history.history['loss'], label='Training Loss')
plt.plot(x, history.history['val_loss'], label='Validation Loss')
plt.plot(x, history.history['accuracy'], label='Training Accuracy')
plt.plot(x, history.history['val_accuracy'], label='Validation Accuracy')

plt.xlim(0, 60)
plt.ylim(0, 20)
plt.ylabel('Val_Loss')
plt.xlabel('Loss')
plt.legend()

plt.tight_layout()
plt.title(f'Model Loss')
plt.savefig('backend/Histroy_TrainVal')
plt.show()
plt.close()

score = model.evaluate(testImages, onehotTest, verbose=0)

print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])

#model.save('./model.keras')

'''
pred = model.predict(testImgs.reshape(testImgs.shape[0], 1, testImgs.shape[1], testImgs.shape[1], 1), bacth_size=13)
pred = np.argmax(pred, axis=1)[:5]
label = np.argmax(onehotTest, axis=1)[:5]

print('Prediction: ', pred)
print('Actual Label: ', label)
'''
