#!/usr/bin/env python3
""" 0. Markov Chain """


import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain being in a particular
    state after a specified number of iterations:
    """

    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if type(s) is not np.ndarray or len(s.shape) != 2:
        return None
    if type(t) is not int or t < 0:
        return None
    if P.shape[0] != P.shape[1] or s.shape[0] != P.shape[0]:
        return None

    for _ in range(t):
        s = np.matmul(s, P)

    return s
