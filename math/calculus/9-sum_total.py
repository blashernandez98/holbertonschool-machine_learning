#!/usr/bin/env python3
""" Task 9 module """


def summation_i_squared(n):
    """ Returns sum of squares until n """
    if type(n) is not int or n < 1:
        return None
    return int((n*(n+1)*(2*n+1))/6)
