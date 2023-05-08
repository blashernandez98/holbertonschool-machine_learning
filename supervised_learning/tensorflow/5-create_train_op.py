#!/usr/bin/env python3
""" Task 5 module """


import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Creates and returns training op tensor
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op
