#!/usr/bin/env python3
""" Exponential module """


class Exponential():
    """ Class representing Exponential distribution """

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
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, k):
        """ Returns PDF of @k """
        e = 2.7182818285
        if k is None or k < 0:
            return 0
        return (self.lambtha * (e ** (-self.lambtha * k)))

    def cdf(self, k):
        """ Returns CDF of @k """
        e = 2.7182818285
        if k is None or k < 0:
            return 0
        return (1 - (e ** (-self.lambtha * k)))
