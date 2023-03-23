#!/usr/bin/env python3
""" Task 3 module """


def matrix_transpose(matrix: list):
    """ Returns the transpose of a 2D matrix """

    out = [[] for row in matrix]
    for row in matrix:
        for col_idx, col in enumerate(row):
            out[col_idx].append(col)

    return out
