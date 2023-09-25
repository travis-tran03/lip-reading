import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras.optimizers import Adam
from network.architecture import neuralNetwork
from allData.resizing import crop
import os

import cv2 as cv

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import pandas as pd

import allData.data as dt

#phrases = ["stop navigation", "excuse me", "i am sorry", "thank you", "good bye", "i love this game", "nice to meet you", "you are welcome", "how are you", "have a good time"]
phrases = ["stop navigation", "excuse me", "i am sorry", "stop navigation", "good bye", "stop navigation", "stop navigation", "you are welcome", "stop navigation", "have a good time"]
#phrases = ["stop navigation", "excuse me", "i am sorry", "thank you", "good bye", "i love this game"]
words = ["begin", "choose", "connection", "navigation", "next", "previous", "start", "stop", "hello", "wed"]

images, labels = dt.getAllFolders()

croppedImgs = crop(images)

imagesList = [np.array(img) for img in croppedImgs]

finalImages = np.stack(imagesList)

labels = np.array(labels)

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
        


def load_images(folder, array):
    for filename in os.listdir(folder):

        if ("depth" in filename):
            break

        img = mpimg.imread(os.path.join(folder, filename))

        if img is not None:
            array.append(img)

       

    return array

test = np.array_split(finalImages, 10)
test2 = np.array_split(labels, 10)
trainImgs = test[0]
testImgs = test[1]
trainLabels = test2[0]
testLabels = test2[1]

model = neuralNetwork()

optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

def oneHotEncode(labels):
    tempPhrases = labels[:]
    unique = np.unique(tempPhrases)

    mapping = {}
    usedNums = np.arange(1, 11)
    for x in range(len(tempPhrases)):
        ranNum = np.random.choice(usedNums, size=1)
        np.delete(usedNums, np.where(usedNums == ranNum[0]))
        mapping[tempPhrases[x]] = ranNum[0]

    for x in range(len(tempPhrases)):
        tempPhrases[x] = mapping[tempPhrases[x]]

    onehot = to_categorical(tempPhrases, num_classes=10)

    onehot = onehot.reshape(len(onehot), 1, 10)

    return onehot

onehotTrain = oneHotEncode(trainLabels)
onehotTest = oneHotEncode(testLabels)

history = model.fit(trainImgs.reshape(len(trainImgs), 1, 76, 76, 1), onehotTrain, epochs=10, batch_size=13, callbacks=[CustomCallBack(trainImgs.reshape(-1, 1, 76, 76, 1), onehotTrain, 'Lip Reading')])

score = model.evaluate(testImgs.reshape(len(testImgs), 1, 76, 76, 1), onehotTest, verbose=0)

print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])

pred = model.predict(testImgs.reshape(len(testImgs), 1, 76, 76, 1))
pred = np.argmax(pred, axis=1)[:5]
label = np.argmax(onehotTest, axis=1)[:5]

print('Prediction: ', pred)
print('Actual Label: ', label)

