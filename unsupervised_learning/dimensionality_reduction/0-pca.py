#!/usr/bin/env python3
""" 0. PCA """

import numpy as np


def pca(X, ndim):
    """
    Function that performs PCA on a dataset
    """
    X_m = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X_m)
    W = vh.T
    Wr = W[:, :ndim]
    T = np.matmul(X_m, Wr)
    return T
