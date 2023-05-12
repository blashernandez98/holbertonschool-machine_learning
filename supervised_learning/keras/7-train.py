#!/usr/bin/env python3
""" Task 7 module """


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
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

    @learning_rate_decay is a boolean that indicates whether
    learning rate decay should be used

    @alpha is the initial learning rate

    """
    callbacks = []

    if early_stopping and validation_data:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)
        callbacks.append(early_stop)
    if learning_rate_decay and validation_data:
        def scheduler(epoch):
            """ Scheduler function """
            return alpha / (1 + decay_rate * epoch)
        learning_rate = K.callbacks.LearningRateScheduler(scheduler,
                                                          verbose=1)
        callbacks.append(learning_rate)
    else:
        callbacks = None

    hist_obj = network.fit(x=data,
                           y=labels,
                           epochs=epochs,
                           batch_size=batch_size,
                           verbose=verbose,
                           shuffle=shuffle,
                           validation_data=validation_data,
                           callbacks=callbacks)
    return hist_obj
