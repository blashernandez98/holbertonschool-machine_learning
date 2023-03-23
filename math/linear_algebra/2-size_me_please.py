#!/usr/bin/env python3

def matrix_shape(matrix: list):
    out = []
    while type(matrix[0]) is list:
        out.append(len(matrix))
        matrix = matrix[0]

    out.append(len(matrix))
    return out
