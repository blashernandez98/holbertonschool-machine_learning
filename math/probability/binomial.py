#!/usr/bin/env python3
""" Binomial distribution module """


class Binomial():
    """ Class representing a Binomial Distribution """

    def __init__(self, data=None, n=1, p=0.5):
        """ Custom init method """

        if data is None:
            if n < 0:
                raise ValueError("n must be a positive value")
            elif p <= 0 or p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p
        elif type(data) is not list:
            raise TypeError("data must be a list")
        elif len(data) < 2:
            raise ValueError("data must contain multiple values")
        else:
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / (len(data))
            self.n = round(mean / (-(variance/mean) + 1))
            self.p = mean / self.n
