#!/usr/bin/env python3
""" Task 13 module """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ Returns concat of two matrices on specific axis """
    return np.concatenate((mat1, mat2), axis=axis)
