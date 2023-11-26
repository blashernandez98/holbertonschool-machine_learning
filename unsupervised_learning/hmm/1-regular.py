#!/usr/bin/env python3
""" 1. Regular Chains """


import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular markov chain
    """

    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None

    n = P.shape[0]

    if np.any(P <= 0):
        return None
    if np.any(P >= 1):
        return None
    if not np.all(np.isclose(np.sum(P, axis=1), 1)):
        return None

    s = np.ones((1, n)) / n
    t = 1

    while True:
        s_prev = s
        s = np.matmul(s, P)
        if np.all(np.isclose(s, s_prev)):
            return s
        if t == 1000:
            return None
        t += 1
