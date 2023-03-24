#!/usr/bin/env python3
""" Task 100 module """


def np_slice(matrix, axes={}):
    """ Returns slice of matrix along @axes """
    slices = [slice(None)] * matrix.ndim
    for axis, slaice in axes.items():
        slices[axis] = slice(*slaice)
    return matrix[tuple(slices)]
