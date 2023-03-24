#!/usr/bin/env python3
""" Task 12 module """


def np_elementwise(mat1, mat2):
    """ Returns sum/dif/mul/div of 2 matrices """

    sum = mat1.__add__(mat2)
    dif = mat1.__sub__(mat2)
    mul = mat1.__mul__(mat2)
    div = mat1.__truediv__(mat2)

    return (sum, dif, mul, div)
