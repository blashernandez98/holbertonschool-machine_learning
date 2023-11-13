#!/usr/bin/env python3
""" 0. PCA """

import numpy as np


def pca(X, var=0.95):
    """
    Function that performs PCA on a dataset
    """
    u, s, vh = np.linalg.svd(X)
    total = np.cumsum(s)
    total /= total[-1]
    r = np.argwhere(total >= var)[0, 0]
    return vh[:r + 1].T
