import numpy as np
import tensorflow as tf
from tensorflow import nn # nn = NeuralNetwork
from keras import layers, models, Input
from keras.layers import Conv2D, Conv3D, ReLU, MaxPool2D, Dense, Flatten, Activation, Dropout, BatchNormalization


# image = single image
def neuralNetwork():

    model = models.Sequential([
        Input(shape= (8, 8, 3)),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2)),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
        Dropout(0.3),

        Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2)),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2)),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size=(2, 2), strides=(1, 1)),
        Dropout(0.3),

        Flatten(),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        BatchNormalization(),

        Dense(20, activation='softmax')

    ]) # Using Convolutional NN Base

    # Sample (WIP Using Keras)
    # layers.Conv2d
    # layers.relu => some activation function (relu is most used)
    # .layers.maxpool
    # *REPEAT*
    # FULLY CONNECTED LAYER
    # layers.flatten => FLATTEN
    # layers.softmax
    # outputs => DENSE
    model.summary()
    return model

model = neuralNetwork()

model.compile()
