#!/usr/bin/env python3
""" Task 7 module """


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenates two matrices along a specific axis """
    if len(mat1) == 0 or len(mat2) == 0:
        return None
    if axis == 0:
        for (row1, row2) in zip(mat1, mat2):
            if len(row1) != len(row2):
                return None
        return mat1 + mat2

    elif axis == 1:
        new = []
        if len(mat1) != len(mat2):
            return None
        for i, (row1, row2) in enumerate(zip(mat1, mat2)):
            new.append([])
            new[i] = row1 + row2
        return new
    return None
