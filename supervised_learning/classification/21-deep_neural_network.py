#!/usr/bin/env python3
""" Task 21 module """


import numpy as np


def int_validate(param, name):
    """ Validates type and value for @param """
    if type(param) is not int:
        raise TypeError("{} must be an integer".format(name))
    if param < 1:
        raise ValueError("{} must be a positive integer".format(name))


def sigmoid(z):
    """ Sigmoid activation function """
    return (1 / (1 + np.exp(-z)))


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
                or any(map(lambda x: x < 0 or type(x) is not int, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(1, self.__L + 1):
            if i == 1:
                self.__weights["W{}".format(i)] = np.random.randn(
                    layers[i - 1], nx) * np.sqrt(2 / nx)
            else:
                self.__weights["W{}".format(i)] = np.random.randn(
                    layers[i - 1], layers[i - 2]) * np.sqrt(2 / layers[i - 2])
            self.__weights["b" + str(i)] = np.zeros((layers[i - 1], 1))

    @property
    def L(self):
        """ Layers getter """
        return self.__L

    @property
    def cache(self):
        """ Cache getter """
        return self.__cache

    @property
    def weights(self):
        """ Weights getter """
        return self.__weights

    def forward_prop(self, X):
        """ Calculates forward propagation """
        self.__cache["A0"] = X
        for i in range(1, self.L + 1):
            weight = self.weights["W{}".format(i)]
            prev_activation = self.cache["A{}".format(i-1)]
            bias = self.weights["b{}".format(i)]

            zI = np.matmul(weight, prev_activation) + bias
            self.__cache["A{}".format(i)] = sigmoid(zI)

        return self.cache["A{}".format(self.L)], self.cache

    def cost(self, Y, A):
        """ Calculates cost for network """
        one = 1.0000001
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(one - A)) / Y.shape[1]

    def evaluate(self, X, Y):
        """ Evaluates network """

        predictions = self.forward_prop(X)[0]
        return (np.where(predictions > 0.5, 1, 0), self.cost(Y, predictions))

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent """

        dz = 0
        m = Y.shape[1]
        for i in range(self.L, 0, -1):
            W_i, b_i = self.weights["W" + str(i)], self.weights["b" + str(i)]
            A_i = self.cache["A" + str(i)]
            A_prev = self.cache["A" + str(i-1)]

            if i == self.L:
                dz = A_i - Y
            else:
                W_next = self.weights["W" + str(i+1)]
                dz = np.dot(W_next.T, dz) * \
                    (sigmoid(A_i) * (1 - sigmoid(A_i)))

            dw = (1 / m) * np.dot(dz, A_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            self.__weights['W' + str(i)] = W_i - alpha * dw
            self.__weights['b{}'.format(i)] = b_i - alpha * db
