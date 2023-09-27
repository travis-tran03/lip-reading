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

def getF1(imgs, labels):
    basePath = 'C:/Users/Crolw/OneDrive/Documents/GitHub/lip-reading/MIRACL-VC1_all_in_one/'

    for i in range(1, 6):
        if (i == 3):
            continue
        for x in range(1, 11):
            for y in range(1, 11):
                path = os.path.join(basePath, f"F{str(i).zfill(2)}/phrases/{str(x).zfill(2)}/{str(y).zfill(2)}")

                newImgs = dt.load_images(path, [])

                for j in range(len(newImgs)):
                    imgs.append(newImgs[j])
                    labels.append(phrases[x-1])
    
    return imgs, labels

#images, labels = dt.getAllFolders()
#images, labels = dt.allFolders()
images, labels = getF1([], [])

print(len(images))
print(len(labels))

#croppedImgs = crop(images)

#print(len(croppedImgs))


imagesList = [np.array(img) for img in images]

finalImages = np.stack(imagesList)

imagesList = [np.array(label) for label in labels]

labels = np.array(imagesList)

print(finalImages.shape)
print(labels.shape)

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



test = np.array_split(finalImages, 4) # Shape = (10, ~1500, 76, 76, 1)
test2 = np.array_split(labels, 4) # Shape = (10, ~1500, 76, 76, 1)
trainImgs = test[0]
testImgs = test[1]
trainLabels = test2[0]
testLabels = test2[1]

model = neuralNetwork()

optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

def oneHotEncode(type, labels):
    tempPhrases = list(type)
    unique = np.unique(tempPhrases)

    mapping = {}
    for x in range(len(unique)):
        mapping[unique[x]] = x
    '''
    usedNums = np.arange(1, 11)
    for x in range(len(unique)):
        ranNum = np.random.choice(usedNums, size=1)
        usedNums = np.delete(usedNums, np.where(usedNums == ranNum[0]))
        mapping[unique[x]] = ranNum[0]
    '''

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
    print(onehotArr.shape)

    onehot = onehot.reshape(len(onehot), 1, 10)

    onehotArr = onehotArr.reshape(len(onehotArr), 1, 10)

    print(onehotArr.shape)

    return onehotArr

onehotTrain = oneHotEncode(phrases, trainLabels)
onehotTest = oneHotEncode(phrases, testLabels)

history = model.fit(trainImgs.reshape(trainImgs.shape[0], 1, trainImgs.shape[1], trainImgs.shape[1], 1), onehotTrain, epochs=30, batch_size=13, callbacks=[CustomCallBack(trainImgs.reshape(trainImgs.shape[0], 1, trainImgs.shape[1], trainImgs.shape[1], 1), onehotTrain, 'Lip Reading')])

score = model.evaluate(testImgs.reshape(testImgs.shape[0], 1, testImgs.shape[1], testImgs.shape[1], 1), onehotTest, verbose=0)

print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])

pred = model.predict(testImgs.reshape(testImgs.shape[0], 1, testImgs.shape[1], testImgs.shape[1], 1))
pred = np.argmax(pred, axis=1)[:5]
label = np.argmax(onehotTest, axis=1)[:5]

print('Prediction: ', pred)
print('Actual Label: ', label)

