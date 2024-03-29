#!/usr/bin/env python3
""" 0. Initialize K-means """

import numpy as np


def initialize(X, k):
    """
    function that initializes cluster centroids for K-means
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    n, d = X.shape
    min = np.amin(X, axis=0)
    max = np.amax(X, axis=0)
    return np.random.uniform(min, max, (k, d))
