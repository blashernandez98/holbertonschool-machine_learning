#!/usr/bin/env python3
""" Task 4 module """


def add_arrays(arr1, arr2):
    """ Return sum of 2 arrays element-wise """
    if len(arr1) == len(arr2):
        return [a + arr2[idx] for idx, a in enumerate(arr1)]
    else:
        return None
