#!/usr/bin/env python3
""" Task 5 module """


def add_matrices2D(mat1, mat2):
    """ Returns sum of 2D matrices element-wise """
    out = []
    if (len(mat1) != len(mat2)) or (len(mat1[0]) != len(mat2[0])):
        return None
    for row_idx, row in enumerate(mat1):
        out.append([])
        for col_idx, col in enumerate(row):
            out[row_idx].append(col + mat2[row_idx][col_idx])
    return out
