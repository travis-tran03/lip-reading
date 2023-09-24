import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from keras.utils import to_categorical
from network.architecture import neuralNetwork
from allData.resizing import crop
import os

import cv2 as cv

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import pandas as pd

#phrases = ["stop navigation", "excuse me", "i am sorry", "thank you", "good bye", "i love this game", "nice to meet you", "you are welcome", "how are you", "have a good time"]
phrases = ["stop navigation", "excuse me", "i am sorry", "stop navigation", "good bye", "stop navigation", "stop navigation", "you are welcome", "stop navigation", "have a good time"]
#phrases = ["stop navigation", "excuse me", "i am sorry", "thank you", "good bye", "i love this game"]
words = ["begin", "choose", "connection", "navigation", "next", "previous", "start", "stop", "hello", "wed"]

#images = getAllFolders()

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
        plt.savefig('model_train_images/'+self.model_name+"_"+str(epoch))
        plt.close()


def load_images(folder, array):
    for filename in os.listdir(folder):

        if ("depth" in filename):
            break

        img = mpimg.imread(os.path.join(folder, filename))

        if img is not None:
            array.append(img)

       

    return array

imgs = load_images('C:/Users/Crolw/OneDrive/Documents/GitHub/lip-reading/MIRACL-VC1_all_in_one/F01/phrases/01/01', [])

croppedArr = crop(imgs)

imgsList = [np.array(img) for img in croppedArr]

croppedArr = np.stack(imgsList)

test = np.array_split(croppedArr, 2)
trainImgs = test[0]
testImgs = test[1]

model = neuralNetwork()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


tempPhrases = phrases[:]
unique = np.unique(tempPhrases)

mapping = {}
usedNums = np.arange(1, 6)
for x in range(len(tempPhrases)):
    ranNum = np.random.choice(usedNums, size=1)
    np.delete(usedNums, np.where(usedNums == ranNum[0]))
    mapping[tempPhrases[x]] = ranNum[0]

for x in range(len(tempPhrases)):
    tempPhrases[x] = mapping[tempPhrases[x]]

onehot = to_categorical(tempPhrases, num_classes=6)

onehot = onehot.reshape(6, 1, 10)

model.fit(trainImgs.reshape(6, 1, 76, 76, 1), onehot, epochs=100, batch_size=6, callbacks=[CustomCallBack(trainImgs.reshape(6, 1, 76, 76, 1), onehot, 'Lip Reading')])

score = model.evaluate(testImgs.reshape(6, 1, 76, 76, 1), onehot, verbose=0)

print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])

pred = model.predict(testImgs.reshape(6, 1, 76, 76, 1))
pred = np.argmax(pred, axis=1)[:5]
label = np.argmax(onehot, axis=1)[:5]

print('Prediction: ', pred)
print('Actual Label: ', label)

