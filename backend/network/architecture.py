import numpy as np
import tensorflow as tf
from tensorflow import nn # nn = NeuralNetwork
from keras import layers, models, Input
from keras.layers import Conv2D, Conv3D, ReLU, MaxPool2D, Dense, Flatten, Activation, Dropout, BatchNormalization, LSTM, TimeDistributed, Bidirectional


# image = single image
def neuralNetwork():
    '''
    model = models.Sequential([
        Input(shape= (64, 64, 1)),

        TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')),
        TimeDistributed(BatchNormalization()),

        TimeDistributed(MaxPool2D(pool_size=(2, 2))),
        TimeDistributed(Dropout(0.25)),

        TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')),
        TimeDistributed(BatchNormalization()),

        TimeDistributed(MaxPool2D(pool_size=(2, 2))),
        TimeDistributed(Dropout(0.25)),

        TimeDistributed(Conv2D(filters=20, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')),
        TimeDistributed(BatchNormalization()),

        TimeDistributed(Flatten()),

        Bidirectional(LSTM(32, kernel_initializer='Orthogonal', return_sequences=True)),
        Dropout(0.25),

        Dense(20, activation='softmax')
    ])
    '''
    model = models.Sequential([
        Input(shape= (None, 64, 64, 1)),

        TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')),
        BatchNormalization(),

        TimeDistributed(MaxPool2D(pool_size=(2, 2))),
        Dropout(0.25),

        TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')),
        BatchNormalization(),

        TimeDistributed(MaxPool2D(pool_size=(2, 2))),
        Dropout(0.25),

        TimeDistributed(Conv2D(filters=20, kernel_size=(3, 3), padding='same', strides=(2, 2), activation='relu')),
        BatchNormalization(),

        TimeDistributed(Flatten()),

        Bidirectional(LSTM(32, kernel_initializer='Orthogonal', return_sequences=True)),
        Dropout(0.25),

        Dense(20, activation='softmax')
    ])
    # Using Convolutional NN Base

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




