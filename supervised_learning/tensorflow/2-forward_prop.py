#!/usr/bin/env python3
""" Task 2 module """

import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Calculates forward propagation for NN
    Returns output tensor of last layer
    """
    create_layer = __import__('1-create_layer').create_layer

    for i, l in enumerate(layer_sizes):
        if i == 0:
            layer = create_layer(x, l, activations[i])
        else:
            layer = create_layer(layer, l, activations[i])

    return layer
