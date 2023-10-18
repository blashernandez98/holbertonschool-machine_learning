#!/usr/bin/env python3
""" 6. Momentum Upgraded """

import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network in tensorflow using
    the gradient descent with momentum optimization algorithm:

    @loss is the loss of the network
    @alpha is the learning rate
    @beta1 is the momentum weight

    Returns: the momentum optimization operation
    """

    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
