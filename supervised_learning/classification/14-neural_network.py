#!/usr/bin/env python3
""" Task 14 module """

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

    def cost(self, Y, A):
        """ Calculates cost for model """
        one = 1.0000001
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(one - A)) / Y.shape[1]

    def evaluate(self, X, Y):
        """ Evaluates model """
        predictions = self.forward_prop(X)[1]
        return (np.where(predictions >= 0.5, 1, 0), self.cost(Y, predictions))

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Updates Weights and Biases with one step of gradient descent """
        m = X.shape[1]
        # Derivatives of output layer
        dz2 = A2 - Y
        dw2 = (1/m) * np.dot(dz2, A1.T)
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

        # Derivative of hidden layer
        dz1 = np.dot(self.W2.T, dz2) * (A1 * (1 - A1))
        dw1 = (1/m) * np.dot(dz1, X.T)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

        self.__W1 = self.W1 - alpha * dw1
        self.__b1 = self.b1 - alpha * db1
        self.__W2 = self.W2 - alpha * dw2
        self.__b2 = self.b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Train model over @iterations """

        int_validate(iterations, "iterations")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(1, iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.A1, self.A2, alpha)

        return self.evaluate(X, Y)
