#!/usr/bin/env python3
""" Task 9 module """


def summation_i_squared(n):
    import numpy as np
    """ Returns sum of squares until n """
    nums = np.arange(1, n+1)**2
    return sum(nums)
