#!/usr/bin/env pyhthon3
""" Task 16 module """


import numpy as np


def int_validate(param, name):
    """ Validates type and value for @param """
    if type(param) is not int:
        raise TypeError("{} must be an integer".format(name))
    if param < 1:
        raise ValueError("{} must be a positive integer".format(name))


class DeepNeuralNetwork():
    """ Represents a deep neural network """

    def __init__(self, nx, layers):
        """
        @nx is the number of input features
        @layers is a list representing the number
        of nodes in each layer of the network
        """

        int_validate(nx, "nx")
        if type(layers) is not list or len(layers) < 1\
                or any(x < 0 or type(x) is not int for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        layers.insert(0, nx)
        for i in range(1, len(layers)):
            self.weights[f"W{i}"] = np.random.randn(
                layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.weights[f"b{i}"] = np.zeros(shape=(layers[i], 1))
