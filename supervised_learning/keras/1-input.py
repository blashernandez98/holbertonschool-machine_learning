#!/usr/bin/env python3
""" Task 1 module """


from tensorflow import keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Builds keras model """

    inputs = k.Input(shape=(nx,))
    x = inputs
    for i in range(0, len(layers)):
        dense = (k.layers.Dense(units=layers[i], activation=activations[i],
                                kernel_regularizer=k.regularizers.l2(lambtha)))
        x = dense(x)
        # Add dropout layer before all layers except last one
        if i != len(layers) - 1:
            dropout = k.layers.Dropout(rate=1-keep_prob)
            x = dropout(x)
    model = k.Model(inputs=inputs, outputs=x)
    return model
