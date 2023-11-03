import numpy as np
import tensorflow as tf
from tensorflow import nn # nn = NeuralNetwork
from keras import layers, models, Input
from keras.layers import Conv2D, Conv3D, ReLU, MaxPool2D, Dense, Flatten, Activation, Dropout, BatchNormalization, LSTM, TimeDistributed, Bidirectional
from keras.layers import ConvLSTM2D, MaxPool3D
from keras.regularizers import L2

def neuralNetwork(frames, rows, columns, channels):
    '''
    # *  ModelCNN: Convulution Neural Network Model *
    # * Input = (28, 28, 1) => One 28 x 28 Image in Grayscale *
    '''

    modelCNN = models.Sequential()

    # Add Layers
    modelCNN.add(Input(shape=(rows, columns, channels)))
    modelCNN.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    modelCNN.add(MaxPool2D(pool_size=(2, 2), strides=2))
    modelCNN.add(BatchNormalization())

    modelCNN.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    modelCNN.add(MaxPool2D(pool_size=(2, 2), strides=2))
    modelCNN.add(BatchNormalization())

    modelCNN.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    modelCNN.add(MaxPool2D(pool_size=(2, 2), strides=2))

    modelCNN.add(Flatten())

    modelCNN.add(Dropout(0.5))
    
    modelCNN.add(Dense(128, activation='relu'))

    # modelCNN.add(LSTM()) # LSTM Shape = LSTM(NumberOfVideos/AnyNumberCells, input_shape=(NumberOfFrames, Height, Width, Channels))

    modelLSTM = models.Sequential()

    modelLSTM.add(Input(shape=(frames, rows, columns, channels))) # LSTM Shape = (NumberOfVideos, NumberOfFrames, Height, Width, Channels)
    modelLSTM.add(TimeDistributed(modelCNN)) # In => (rows, columns, channels); Out => (None, 128)
    modelLSTM.add(LSTM(256, return_sequences=True, activation='tanh')) # (None, 10, 128)
    modelLSTM.add(Dropout(0.5))
    modelLSTM.add(LSTM(128, activation='tanh'))
    modelLSTM.add(Dropout(0.5))
    modelLSTM.add(Dense(128, activation='relu'))
    modelLSTM.add(Dense(10, activation='softmax'))

    modelCNN.summary()
    modelLSTM.summary()
    return modelLSTM