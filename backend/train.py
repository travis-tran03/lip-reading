import os

import numpy as np

from keras.callbacks import Callback
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
#phrases = np.array(["stop navigation", "excuse me", "i am sorry", "stop navigation", "good bye", "stop navigation", "stop navigation", "you are welcome", "stop navigation", "have a good time"])
#phrases = ["stop navigation", "excuse me", "i am sorry", "thank you", "good bye", "i love this game"]
words = np.array(["begin", "choose", "connection", "navigation", "next", "previous", "start", "stop", "hello", "wed"])

class CustomCallBack(Callback):
    def __init__(self, x_test, y_test, model_name):
        self.x_test = x_test
        self.y_test = y_test
        self.model_name = model_name
        
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_test)
        fig, ax = plt.subplots(figsize=(8,4))
        plt.scatter(self.y_test, y_pred, alpha=0.6, 
            color='#FF0000', lw=1, ec='black')
        
        lims = [0, 5]

        plt.plot(lims, lims, lw=1, color='#0000FF')
        plt.ticklabel_format(useOffset=False, style='plain')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(lims)
        plt.ylim(lims)

        plt.tight_layout()
        plt.title(f'Prediction Visualization Keras Callback - Epoch: {epoch}')
        plt.savefig('backend/model_train_images/'+self.model_name+"_"+str(epoch))
        plt.close()

        plt.scatter(epoch, logs['loss'], alpha=0.6, 
            color='#FF0000', lw=1, ec='black')
        plt.plot([0, 40], [0, 7], color='#0000FF')
        plt.ticklabel_format(useOffset=False, style='plain')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim([0, 40])
        plt.ylim([0, 7])

        plt.tight_layout()
        plt.title(f'Loss Visualization Keras Callback - Epoch: {epoch}')
        plt.savefig('backend/model_loss/'+self.model_name+"_"+str(epoch))
        plt.close()

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

                if (x > 5 and x < 9):
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

    onehot = onehot.reshape(len(onehot), 1, 10)

    onehotArr = onehotArr.reshape(len(onehotArr), 1, 10)

    return onehotArr
'''
for frame in notCropped:
    cv.imshow('cropImg', frame['img'])
    cv.imshow('otherCrop', frame['cropImg'])
    cv.waitKey(0)
'''
'''
for frame in notCropped:
    cv.imshow(f'img at index: {frame["index"]}', frame['img'])
    cv.imshow(f'greyImg at index: {frame["index"]}', frame['greyImg'])
    cv.imshow(f'cropImg at index: {frame["index"]}', frame['cropImg'])
    cv.waitKey(0)
'''
def convertToNumpy(arr):
    arrList = [np.array(item) for item in arr]

    finalList = np.stack(arrList)  

    return finalList

#images, labels = dt.getAllFolders()
#images, labels = dt.allFolders()
images, labels, notCropped, testImages, testLabel, valData, valLabels = getData()

finalImages = convertToNumpy(images)

labels = convertToNumpy(labels)

testImages = convertToNumpy(testImages)

testLabel = convertToNumpy(testLabel)

valData = convertToNumpy(valData)

valLabels = convertToNumpy(valLabels)

model = neuralNetwork()

optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

onehotTrain = oneHotEncode(phrases, labels)
onehotTest = oneHotEncode(phrases, testLabel)
onehotVal = oneHotEncode(phrases, valLabels)

#finalImages = finalImages.reshape(finalImages.shape[0], 1, finalImages.shape[1], finalImages.shape[1], 1)

history = model.fit(finalImages.reshape(finalImages.shape[0], 1, finalImages.shape[1], finalImages.shape[1], 1), onehotTrain, epochs=10, batch_size=13,
                    callbacks=[CustomCallBack(finalImages.reshape(finalImages.shape[0], 1, finalImages.shape[1], finalImages.shape[1], 1), onehotTrain, 'Lip Reading')],
                    validation_data=(valData.reshape(valData.shape[0], 1, valData.shape[1], valData.shape[1], 1), onehotVal),
                    validation_batch_size=13, shuffle='batch_size')


print(f'Loss: {history.history["loss"]}')
print(f'Val_Loss: {history.history["val_loss"]}')
print(f'Accuracy: {history.history["accuracy"]}')
print(f'Val_Accuracy: {history.history["val_accuracy"]}')

plt.plot(history.history['loss'], history.history['val_loss'])
plt.xlim(0, 12)
plt.ylim(0, 12)
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.tight_layout()
plt.title(f'Model Loss')
plt.savefig('backend/Histroy_TrainVal')
plt.show()
plt.close()

score = model.evaluate(testImages.reshape(testImages.shape[0], 1, testImages.shape[1], testImages.shape[1], 1), onehotTest, verbose=0)

print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])

'''
pred = model.predict(testImgs.reshape(testImgs.shape[0], 1, testImgs.shape[1], testImgs.shape[1], 1), bacth_size=13)
pred = np.argmax(pred, axis=1)[:5]
label = np.argmax(onehotTest, axis=1)[:5]

print('Prediction: ', pred)
print('Actual Label: ', label)
'''
