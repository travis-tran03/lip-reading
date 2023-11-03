import numpy as np
import tensorflow as tf
from tensorflow import nn # nn = NeuralNetwork
from keras import layers, models, Input
from keras.layers import Conv2D, Conv3D, ReLU, MaxPool2D, Dense, Flatten, Activation, Dropout, BatchNormalization, LSTM, TimeDistributed, Bidirectional
from keras.layers import ConvLSTM2D, MaxPool3D
from keras.regularizers import L2

# image = single image

def neuralNetwork(seqLen, imgHeight, imgWidth):


    model = models.Sequential([
        ConvLSTM2D(filters=4, kernel_size=(3, 3), strides=(2, 2), activation='relu', recurrent_dropout=0.7, return_sequences=True, 
                   input_shape=(seqLen, imgHeight, imgWidth, 3), data_format='channels_last'),
        
        BatchNormalization(),
        TimeDistributed(MaxPool2D(pool_size=(2, 2), padding='same')),
        Dropout(0.7),
        
        ConvLSTM2D(filters=8, kernel_size=(3, 3), strides=(2, 2), activation='relu', recurrent_dropout=0.7, return_sequences=True, 
                    data_format='channels_last'),
        BatchNormalization(),
        TimeDistributed(MaxPool2D(pool_size=(2, 2), padding='same')),
        Dropout(0.7),

        ConvLSTM2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu', recurrent_dropout=0.7, return_sequences=True, 
                    data_format='channels_last'),
        BatchNormalization(),
        TimeDistributed(MaxPool2D(pool_size=(2, 2), padding='same')),
        Dropout(0.7),


        Flatten(),
        Dense(10, activation='softmax')
    ])

    '''
    model = models.Sequential([
        Input((None, 91, 91, 1)),
        #input layer of neural network with 91 by 91 image and is a grayscale image with 1 channel

        TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=L2(0.01), bias_regularizer=L2(0.01), padding='same', strides=2, activation='relu')),
        TimeDistributed(BatchNormalization()),
        #2d convolutional layer (stack of filtering images) for neural network with 32 filters and a 3x3 kernel size. 
        #padding ensures that the output has the same dimensions as the input 
        #relu makes every negative value 0 for easier data managing 

        TimeDistributed(MaxPool2D(pool_size=(2, 2))),
        TimeDistributed(Dropout(0.50)),
        #maxpooling downscales the image and extracts the max value according to the filter  
        #0.25 sets the dropout rate meaning a percentage of the inputs will be removed. 

        #2d convolutional layer (stack of filtering images) for neural network with 32 filters and a 3x3 kernel size. 
        #padding ensures that the output has the same dimensions as the input 
        #relu makes every negative value 0 for easier data managing 

        TimeDistributed(Conv2D(filters=10, kernel_size=(3, 3), kernel_regularizer=L2(0.01), bias_regularizer=L2(0.01), padding='same', strides=2, activation='relu')),
        TimeDistributed(BatchNormalization()),

        TimeDistributed(MaxPool2D(pool_size=(2, 2))),
        TimeDistributed(Dropout(0.50)),

        TimeDistributed(Flatten()),
        #Flatten turns the 2D shape input that we have and converts it into a 1D shape (basically a long line of pixels)



       Bidirectional(LSTM(8, kernel_initializer='Orthogonal', kernel_regularizer=L2(0.01), bias_regularizer=L2(0.01), recurrent_regularizer=L2(0.01), return_sequences=True)),
        Dropout(0.60),
        #32 LSTM cells help process information in our data

        Dense(10, activation='softmax')
        #10 dense neurons in each layer and is connected to previous layers. 

    ])
    '''
    '''
    model = models.Sequential([

        Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(76, 76, 1)),
        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),

        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
        BatchNormalization(),

        TimeDistributed(Flatten()),

        Bidirectional(LSTM(32, kernel_initializer='Orthogonal', return_sequences=True)),
        Dropout(0.25),

        Dense(6, activation='softmax')
    ])
    '''

    model.summary()
    return model




