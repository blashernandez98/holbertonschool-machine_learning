#!/usr/bin/env python3
""" Task 6. Transition Layer """


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in:
    Densely Connected Convolutional Networks (2018)
    (https://arxiv.org/pdf/1608.06993.pdf)

    - X is the output from the previous layer
    - nb_filters is an integer representing the number of filters in X
    - compression is the compression factor for the transition layer
    - Your code should implement compression as used in DenseNet-C
    - All weights should use he normal initialization
    - All convolutions should be preceded by Batch Normalization and a
        rectified linear activation (ReLU), respectively
    - Returns: The output of the transition layer and the
        number of filters within the output, respectively
    """
    init = K.initializers.he_normal()
    batch_norm = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch_norm)
    nb_filters = int(nb_filters * compression)
    conv = K.layers.Conv2D(filters=nb_filters,
                           kernel_size=(1, 1),
                           padding='same',
                           kernel_initializer=init)(activation)
    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                         strides=(2, 2),
                                         padding='valid')(conv)
    return avg_pool, nb_filters
