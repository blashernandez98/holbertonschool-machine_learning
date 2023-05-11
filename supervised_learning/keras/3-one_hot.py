#!/usr/bin/env python3
""" Task 3 module """


import tensorflow.keras as k


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix:

    The last dimension of the one-hot matrix must be the number of classes
    Returns: the one-hot matrix
    """

    return k.utils.to_categorical(labels, num_classes=classes)
