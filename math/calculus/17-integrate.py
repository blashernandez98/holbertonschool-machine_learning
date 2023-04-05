#!/usr/bin/env python3
""" Task 17 module """


def poly_integral(poly, c=0):
    """ Return derivative of poly """
    if type(poly) is not list or len(poly) == 0:
        return None
    expo = 1
    out = [c]
    for term in poly:
        out.append(term / expo)
        expo += 1
    return out
