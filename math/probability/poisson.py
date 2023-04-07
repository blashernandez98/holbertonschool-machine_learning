#!/usr/bin/env python3
""" Task 0 Module """


class Poisson():
    """ Class representing Poisson distribution """

    def __init__(self, data=None, lambtha=1.):
        """ Custom init method """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = lambtha
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)
