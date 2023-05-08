#!/usr/bin/env python3
""" Task 0 module """


import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds keras model """

    model = k.Sequential()
    model.add(k.layers.Dense(units=layers[0], input_shape=(nx,),
              kernel_regularizer=k.regularizers.l2(lambtha),
              activation=activations[0]))
    for i in range(1, len(layers)):
        model.add(k.layers.Dropout(rate=1-keep_prob))
        model.add(k.layers.Dense(units=layers[i], activation=activations[i],
                  kernel_regularizer=k.regularizers.l2(lambtha)))
    return model
