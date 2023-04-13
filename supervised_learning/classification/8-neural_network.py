#!/usr/bin/env python3
""" Task 8 module """

import numpy as np


def int_validate(param, name):
    """ Validates type and value for @param """
    if type(param) is not int:
        raise TypeError("{} must be an integer".format(name))
    if param < 1:
        raise ValueError("{} must be a positive integer".format(name))


class NeuralNetwork():
    """
    Neural Network class with one hidden layer
    with @nx input nodes and @nodes nodes in hidden layer
    """

    def __init__(self, nx, nodes):
        """ Custom init method """

        int_validate(nx, "nx")
        int_validate(nodes, "nodes")

        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros(shape=(nodes, 1))
        self.A1 = 0

        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
