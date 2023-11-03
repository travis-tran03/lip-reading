import os
from dotenv import load_dotenv

import numpy as np

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.optimizers import Adam

import cv2 as cv

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import pandas as pd

#from network.architecture import neuralNetwork
from network.arch import neuralNetwork
from allData.resizing import crop
import allData.data as dt

df = pd.DataFrame(columns=['Phrases', 'Words'])

phrases = ["stop navigation", "excuse me", "i am sorry", "thank you", "good bye", "i love this game", "nice to meet you", "you are welcome", "how are you", "have a good time"]
words = np.array(["begin", "choose", "connection", "navigation", "next", "previous", "start", "stop", "hello", "wed"])

df['Phrases'] = phrases
df['Words'] = words

SEQUENCE_LENTH = 8

load_dotenv('backend/.env')

basePath = os.getenv("FOLDERPATH")

def data():
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

                newImgs, lipArray, leftShiftArr, leftLipArr, upShiftArr, upLipArr, newNotCropped = dt.load_images(path, [], [])
                ''' PORBLEM ; SOLUTION => SPLIT IMAGES INTO THEIR SEPERATE VIDEOS '''
                #print('newImg Len', len(newImgs))

                def frameSkip(arr):
                    frames = []

                    skipFrames = max(int(len(arr) / SEQUENCE_LENTH), 1)

                    for frameCount in range(SEQUENCE_LENTH):
                        index = frameCount*skipFrames

                        if index > len(arr)-1:
                            break


                        frame = arr[index]

                        frames.append(frame)

                    if len(frames) == SEQUENCE_LENTH:
                        if (y > 7 and y < 10):
                            valData.append(frames)
                            valLabel.append(phrases[x-1])
                        elif (y == 10):
                            testData.append(frames)
                            testLabel.append(phrases[x-1])
                        else:
                            imgs.append(frames)
                            labels.append(phrases[x-1])

                frameSkip(newImgs)
                frameSkip(lipArray)
                frameSkip(leftShiftArr)
                frameSkip(upShiftArr)
                frameSkip(leftLipArr)
                frameSkip(upLipArr)
                    
                
                for k in range(len(newNotCropped)):
                    notCropped.append(newNotCropped[k])

    return imgs, labels, notCropped, testData, testLabel, valData, valLabel

def getData():
    imgs = []
    labels = []

    notCropped = []

    testData = []
    testLabel = []

    valData = []
    valLabel = []

    for i in range(1, 5):
        if (i == 3):
            continue
        for x in range(1, 11):
            for y in range(1, 11):
                path = os.path.join(basePath, f"F{str(i).zfill(2)}/phrases/{str(x).zfill(2)}/{str(y).zfill(2)}")

                newImgs, newNotCropped = dt.load_images(path, [], [])



                if (y == 8 or y == 9):
                    for j in range(len(newImgs)):
                        valData.append(newImgs[j])
                        valLabel.append(phrases[x-1])
                elif (y == 10):
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

trainLabels = convertToNumpy(trainLabels)

testImages = convertToNumpy(testImages)

testLabels = convertToNumpy(testLabels)

validationImages = convertToNumpy(validationImages)

validationLabels = convertToNumpy(validationLabels)

model = neuralNetwork(trainImages.shape[1], trainImages.shape[2], trainImages.shape[3], trainImages.shape[4])

optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

onehotTrain = oneHotEncode(phrases, trainLabels)
onehotTest = oneHotEncode(phrases, testLabels)
onehotValidation = oneHotEncode(phrases, validationLabels)

earlyStoppiingCallback = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)

history = model.fit(trainImages, onehotTrain, epochs=50, batch_size=16,
                    callbacks=[earlyStoppiingCallback],
                    validation_data=(validationImages, onehotValidation),
                    validation_batch_size=16, shuffle=True)

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
