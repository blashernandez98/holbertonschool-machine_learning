#!/usr/bin/env python3
""" Task 12 module """
import numpy as np


def np_elementwise(mat1, mat2):
    """ Returns sum/dif/mul/div of 2 matrices """

    sum = np.add(mat1, mat2)
    dif = np.subtract(mat1, mat2)
    mul = np.multiply(mat1, mat2)
    div = np.divide(mat1, mat2)

    return (sum, dif, mul, div)
