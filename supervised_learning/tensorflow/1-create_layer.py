#!/usr/bin/env python3
""" Task 1 module """


import tensorflow as tf


def create_layer(prev, n, activation):
    """ Creates and returns layer """
    initializer = tf.contrib.layers.\
        variance_scaling_initializer(mode="FAN_AVG")

    # Create the layer:
    layer = tf.layers.Dense(units=n, name="layer", activation=activation,
                            kernel_initializer=initializer)

    return layer(prev)
