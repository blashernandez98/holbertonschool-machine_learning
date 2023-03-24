#!/usr/bin/env python3
""" Task 8 module """


def mat_mul(mat1, mat2):
    """ Returns matrix multiplication """
    if not mat1 or not mat2 or len(mat1[0]) != len(mat2):
        return None

    new = []

    for i in range(len(mat1)):
        new.append([])
        for k in range(len(mat2[0])):
            sum = 0
            for j in range(len(mat1[i])):
                sum += (mat1[i][j] * mat2[j][k])
            new[i].append(sum)
    return new
