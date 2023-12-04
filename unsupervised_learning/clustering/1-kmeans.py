#!/usr/bin/env python3
""" 1. K-means """


import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)
    C = np.random.uniform(min, max, (k, d))

    for i in range(iterations):
        C_copy = np.copy(C)
        distances = np.linalg.norm(X - C[:, np.newaxis], axis=2)
        clusters = np.argmin(distances, axis=0)
        for j in range(k):
            if len(X[clusters == j]) == 0:
                C[j] = np.random.uniform(min, max, (1, d))
            else:
                C[j] = np.mean(X[clusters == j], axis=0)
        distances = np.linalg.norm(X - C[:, np.newaxis], axis=2)
        clusters = np.argmin(distances, axis=0)
        if (C_copy == C).all():
            break

    return C, clusters
