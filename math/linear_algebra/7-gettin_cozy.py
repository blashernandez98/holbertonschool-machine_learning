#!/usr/bin/env python3
""" Task 7 module """


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenates two matrices along a specific axis """
    new = []
    if axis == 0:
        return mat1 + mat2

    for i, (row1, row2) in enumerate(zip(mat1, mat2)):
        new.append([])
        new[i] = row1 + row2
    return new
