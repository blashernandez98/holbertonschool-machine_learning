#!/usr/bin/env python3
""" Task 1 module """

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
