#!/usr/bin/env python3
""" Task 9 module """

import numpy as np


def summation_i_squared(n):
    """ Returns sum of squares until n """
    nums = np.arange(1, n+1)**2
    return sum(nums)
