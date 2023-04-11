#!/usr/bin/env python3
""" Task 2 module """

import numpy as np


class Neuron():
    """ Neuron for binary classification """

    def __init__(self, nx):
        """ Custom init method """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter for W """
        return self.__W

    @property
    def b(self):
        """ Getter for b """
        return self.__b

    @property
    def A(self):
        """ Getter for A """
        return self.__A

    def forward_prop(self, X):
        """ Forward propagation method """

        if X.shape[0] != self.W.shape[1]:
            raise ValueError("X should have shape (nx, m)")

        z = np.dot(self.W, X) + self.b
        self.__A = 1 / (1 + np.exp(-z))
        return self.A
