#!/usr/bin/env python3
""" Task 0. Likelihood """

import numpy as np


def factorial(n):
    """
    Calculates the factorial of a number
    """
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects.
    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical probabilities
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray:
        raise TypeError("P must be a 1D numpy.ndarray")
    if len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    # Calculate likelihood
    n_fact = factorial(n)
    x_fact = factorial(x)
    n_x_fact = factorial(n - x)
    combination = n_fact / (x_fact * n_x_fact)
    likelihood = combination * (P ** x) * ((1 - P) ** (n - x))
    return likelihood
