#!/usr/bin/env python3
""" Task 3 module """


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    @y is a placeholder for the labels of the input data
    @y_pred is a tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction
    """

    correct = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
