#!/usr/bin/env python3
""" Task 10 module """


def poly_derivative(poly):
    """ Return derivative of poly """
    if len(poly) == 1:
        return [0]
    expo = 0
    out = []
    for term in poly:
        out.append(term * expo)
        expo += 1
    return out[1:]
