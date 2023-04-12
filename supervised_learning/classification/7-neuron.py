#!/usr/bin/env python3
""" Task 7 module """

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

    def cost(self, Y, A):
        """ Calculates cost for model """
        one = 1.0000001
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(one - A)) / Y.shape[1]

    def evaluate(self, X, Y):
        """ Evaluates neuron """
        predictions = self.forward_prop(X)
        return (np.where(predictions > 0.5, 1, 0), self.cost(Y, predictions))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Updates W and B with one step of gradient descent """
        error = A - Y
        m = X.shape[1]
        dw = np.dot(error, X.T) / m
        db = np.sum(error) / m
        self.__W -= alpha * dw
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """ Trains neuron @iterations times """
        import matplotlib.pyplot as plt

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if any((verbose, graph)):
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        iters = []
        costs = []

        for i in range(iterations + 1):
            
            A = self.forward_prop(X)
            cost = self.cost(Y, A)

            iters.append(i)
            costs.append(cost)

            if (verbose and i % step == 0) or (i == iterations):
                print(f"Cost after {i} iterations: {cost}")

            self.gradient_descent(X, Y, A, alpha)
        
        plt.plot(iters, costs)
        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.title("Training Cost")

        return self.evaluate(X, Y)
