#!/usr/bin/env python3
""" Task 8 module """

import numpy as np


def int_validate(param, name):
    """ Validates type and value for @param """
    if type(param) is not int:
        raise TypeError(f"{name} must be an integer")
    if param < 1:
        raise ValueError(f"{name} must be a positive integer")


class NeuralNetwork():
    """
    Neural Network class with one hidden layer
    with @nx input nodes and @nodes nodes in hidden layer
    """

    def __init__(self, nx, nodes):
        """ Custom init method """

        int_validate(nx, "nx")
        int_validate(nodes, "nodes")

        self.W1 = np.random.normal(size=(nx, nodes))
        self.b1 = np.zeros(size=(1, nodes))
        self.A1 = 0

        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
