#!/usr/bin/env python3
""" Task 17 module """


def poly_integral(poly, C=0):
    """ Return integral of poly """
    if type(poly) is not list or len(poly) == 0 or type(C) is not int:
        return None
    expo = 1
    out = [C]
    for term in poly:
        new_term = term / expo
        if new_term.is_integer():
            out.append(int(new_term))
        else:
            out.append(new_term)
        expo += 1
    return out
