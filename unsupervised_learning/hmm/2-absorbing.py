#!/usr/bin/env python3
""" 2. Absorbing Chains """


import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing
    """

    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False

    n = P.shape[0]

    if np.any(P < 0) or np.any(P > 1):
        return False
    if not np.all(np.isclose(np.sum(P, axis=1), 1)):
        return False

    if np.all(np.diag(P) == 1):
        return True

    if np.any(np.diag(P) == 1):
        absorbing_states = np.where(np.diag(P) == 1)[0]
        for state in absorbing_states:
            if np.all(P[state, :] == 0):
                return False
        return True
    return False
