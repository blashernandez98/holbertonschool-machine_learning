#!/usr/bin/env python3
""" Task 2 module """


import tensorflow.keras as k


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a keras model with categorical
    crossentropy loss and accuracy metrics:

    network is the model to optimize
    alpha is the learning rate
    beta1 is the first Adam optimization parameter
    beta2 is the second Adam optimization parameter
    Returns: None
    """

    Adam = k.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=Adam, loss="categorical_crossentropy",
                    metrics=[k.metrics.categorical_accuracy])
    return None
