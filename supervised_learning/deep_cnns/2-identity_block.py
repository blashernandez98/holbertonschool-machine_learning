#!/usr/bin/env python3
""" Task 2. Identity Block """


import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in:
    Deep Residual Learning for Image Recognition (2015)
    (https://arxiv.org/pdf/1512.03385.pdf)

    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
    All convolutions inside the block should be
    followed by batch normalization along the channels axis and
    a rectified linear activation (ReLU), respectively.

    All weights should use he normal initialization
    Returns: the activated output of the identity block
    """

    F11, F3, F12 = filters
    init = K.initializers.he_normal()

    conv1 = K.layers.Conv2D(
        filters=F11, kernel_size=(1, 1),
        padding='same', kernel_initializer=init)(A_prev)

    batch_norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation('relu')(batch_norm1)

    conv2 = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3),
        padding='same', kernel_initializer=init)(relu1)
    batch_norm2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu2 = K.layers.Activation('relu')(batch_norm2)

    conv3 = K.layers.Conv2D(
        filters=F12, kernel_size=(1, 1),
        padding='same', kernel_initializer=init)(relu2)
    batch_norm3 = K.layers.BatchNormalization(axis=3)(conv3)

    add = K.layers.Add()([batch_norm3, A_prev])
    output = K.layers.Activation('relu')(add)

    return output
