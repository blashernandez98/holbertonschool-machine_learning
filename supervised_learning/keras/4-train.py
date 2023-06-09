#!/usr/bin/env python3
""" Task 4 module """


import tensorflow.keras as k


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent:

    @network is the model to train

    @data is a numpy.ndarray of shape (m, nx) containing the input data

    @labels is a one-hot numpy.ndarray of shape (m, classes)
    containing the labels of data

    @batch_size is the size of the batch used for mini-batch gradient descent

    @epochs is the number of passes through data
    for mini-batch gradient descent

    @verbose is a boolean that determines if output should be printed

    @shuffle is a boolean that determines whether
    to shuffle the batches every epoch.
    Returns: the History object generated after training the model
    """

    hist_obj = network.fit(x=data,
                           y=labels,
                           epochs=epochs,
                           batch_size=batch_size,
                           verbose=verbose,
                           shuffle=shuffle)
    return hist_obj
