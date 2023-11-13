#!/usr/bin/env python3
""" 3. Posterior """

import numpy as np
intersection = __import__('1-intersection').intersection


def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data:

    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical probabilities
    of developing severe side effects
    Pr is a 1D numpy.ndarray containing the prior beliefs of P
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
    if type(Pr) is not np.ndarray:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate intersection
    intersection_values = intersection(x, n, P, Pr)
    # Calculate marginal probability
    marginal = np.sum(intersection_values)
    # Calculate posterior
    posterior = intersection_values / marginal
    return posterior
