#!/usr/bin/env python3
""" Task 0. Inception Block """


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block as described in
    Going Deeper with Convolutions (2014)
    link: https://arxiv.org/pdf/1409.4842.pdf

    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP:
        F1 is the number of filters in the 1x1 conv
        F3R is the number of filters in the 1x1 conv before the 3x3 conv
        F3 is the number of filters in the 3x3 conv
        F5R is the number of filters in the 1x1 conv before the 5x5 conv
        F5 is the number of filters in the 5x5 conv
        FPP is the number of filters in the 1x1 conv after the max pooling
    All convs inside the inception block should use ReLU activation
    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal()

    conv1 = K.layers.Conv2D(filters=F1,
                            kernel_size=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_initializer=init)(A_prev)

    conv3R = K.layers.Conv2D(filters=F3R,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu',
                             kernel_initializer=init)(A_prev)

    conv3 = K.layers.Conv2D(filters=F3,
                            kernel_size=(3, 3),
                            padding='same',
                            activation='relu',
                            kernel_initializer=init)(conv3R)

    conv5R = K.layers.Conv2D(filters=F5R,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='relu',
                             kernel_initializer=init)(A_prev)

    conv5 = K.layers.Conv2D(filters=F5,
                            kernel_size=(5, 5),
                            padding='same',
                            activation='relu',
                            kernel_initializer=init)(conv5R)

    pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                 strides=(1, 1),
                                 padding='same')(A_prev)

    convFPP = K.layers.Conv2D(filters=FPP,
                              kernel_size=(1, 1),
                              padding='same',
                              activation='relu',
                              kernel_initializer=init)(pool)

    output = K.layers.concatenate([conv1, conv3, conv5, convFPP])

    return output
