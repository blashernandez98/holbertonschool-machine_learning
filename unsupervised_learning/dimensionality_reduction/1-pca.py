#!/usr/bin/env python3
""" 1. PCA v2 """

import numpy as np


def pca(X, ndim):
    """
    Function that performs PCA on a dataset
    """
    X_m = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X_m)
    W = vh[:ndim].T
    T = np.matmul(X_m, W)
    return T
