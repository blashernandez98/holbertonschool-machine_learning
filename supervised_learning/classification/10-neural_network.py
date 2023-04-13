#!/usr/bin/env python3
""" Task 10 module """

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

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros(shape=(nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ Getter for W1 """
        return self.__W1

    @property
    def b1(self):
        """ Getter for b1 """
        return self.__b1

    @property
    def A1(self):
        """ Getter for A1 """
        return self.__A1

    @property
    def W2(self):
        """ Getter for W2 """
        return self.__W2

    @property
    def b2(self):
        """ Getter for b2 """
        return self.__b2

    @property
    def A2(self):
        """ Getter for A2 """
        return self.__A2

    def forward_prop(self, X):
        """ Calculates forward propagation with @X input """

        z1 = np.dot(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + np.exp(-z1))

        z2 = np.dot(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + np.exp(-z2))

        return (self.A1, self.A2)
