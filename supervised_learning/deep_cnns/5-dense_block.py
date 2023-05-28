#!/usr/bin/env python3
""" Task 5. Dense block """


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in:
    Densely Connected Convolutional Networks (2016)
    (https://arxiv.org/pdf/1608.06993.pdf)

    - X is the output from the previous layer
    - nb_filters is an integer representing the number of filters in X
    - growth_rate is the growth rate for the dense block
    - layers is the number of layers in the dense block
    - You should use the bottleneck layers used for DenseNet-B
    - All weights should use he normal initialization
    - All convolutions should be preceded by Batch Normalization and
    a rectified linear activation (ReLU), respectively

    Returns: The concatenated output of each layer within the Dense Block and
    the number of filters within the concatenated outputs, respectively
    """

    init = K.initializers.he_normal()

    for layer in range(layers):
        batch1 = K.layers.BatchNormalization()(X)
        relu1 = K.layers.Activation('relu')(batch1)
        conv1 = K.layers.Conv2D(
            filters=4 * growth_rate, kernel_size=(1, 1),
            padding='same', strides=(1, 1),
            kernel_initializer=init)(relu1)

        batch2 = K.layers.BatchNormalization()(conv1)
        relu2 = K.layers.Activation('relu')(batch2)
        conv2 = K.layers.Conv2D(
            filters=growth_rate, kernel_size=(3, 3),
            padding='same', strides=(1, 1),
            kernel_initializer=init)(relu2)

        X = K.layers.concatenate([X, conv2])
        nb_filters += growth_rate

    return X, nb_filters
