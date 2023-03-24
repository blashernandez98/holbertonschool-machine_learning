#!/usr/bin/env python3
""" Task 7 module """


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenates two matrices along a specific axis """
    if axis == 0:
        return mat1 + mat2
    try:
        for row_idx, row in enumerate(mat1):
            mat1[row_idx].append(*mat2[row_idx])
        return mat1
    except IndexError as e:
        return None
