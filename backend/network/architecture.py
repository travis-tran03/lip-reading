import numpy as np
import tensorflow as tf
from tensorflow import nn # nn = NeuralNetwork
from keras import layers, models, Input


# image = single image
def neuralNetwork(image):

    model = models.Sequential() # Using Convolutional NN Base

    model.add(Input(shape= (8, 8, 3)))
    model.add(layers.Conv2D())

    # Sample (WIP Using Keras)
    # layers.Conv2d
    # layers.relu => some activation function (relu is most used)
    # .layers.maxpool
    # *REPEAT*
    # layers.flatten => FLATTEN
    # layers.softmax
    # outputs => DENSE

neuralNetwork()