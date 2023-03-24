#!/usr/bin/env python3
""" Task 7 module """


def cat_matrices2D(mat1, mat2, axis=0):
    """ Concatenates two matrices along a specific axis """
    if len(mat1) == 0 or len(mat2) == 0:
        return None
    new = []
    if axis == 0:
        for i, (row1, row2) in enumerate(zip(mat1, mat2)):
            if len(row1) == 0 or len(row2) == 0 or len(row1) != len(row2):
                return None
            new.extend(row1)
            new.extend(row2)
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        for i, (row1, row2) in enumerate(zip(mat1, mat2)):
            new.append([])
            new[i] = row1 + row2
    return new
