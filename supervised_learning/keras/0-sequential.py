#!/usr/bin/env python3
""" Task 0 module """


import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras import regularizers


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds keras model """

    model = keras.Sequential()
    model.add(Dense(units=layers[0], input_shape=(nx,),
              kernel_regularizer=regularizers.l2(lambtha)))
    for i in range(1, len(layers)):
        model.add(Dropout(rate=keep_prob))
        model.add(Dense(units=layers[i], activation=activations[i],
                  kernel_regularizer=regularizers.l2(lambtha)))
    return model
