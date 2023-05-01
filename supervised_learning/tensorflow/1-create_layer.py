#!/usr/bin/env python3
""" Task 1 module """


import tensorflow as tf


def create_layer(prev, n, activation):
    """ Creates and returns layer """
    initializer = tf.contrib.layers.\
        variance_scaling_initializer(mode="FAN_AVG")

    W = tf.Variable(initializer([prev.shape[1].value, n]))
    b = tf.Variable(tf.zeros([n]))

    z = tf.matmul(prev, W) + b

    return activation(z)
