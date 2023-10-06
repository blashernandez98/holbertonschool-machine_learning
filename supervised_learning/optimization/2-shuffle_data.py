#!/usr/bin/env python3
""" 2. Shuffle Data """

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    """
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]
