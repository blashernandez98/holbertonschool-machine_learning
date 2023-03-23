#!/usr/bin/env python3
""" Task 2 Module """


def matrix_shape(matrix: list):
    """ Returns shape of matrix """
    out = []
    while type(matrix[0]) is list:
        out.append(len(matrix))
        matrix = matrix[0]

    out.append(len(matrix))
    return out
