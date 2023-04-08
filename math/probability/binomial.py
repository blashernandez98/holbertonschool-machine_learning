#!/usr/bin/env python3
""" Binomial distribution module """


def factorial(n):
    """ Factorial of @n """
    res = 1
    for i in range(1, n+1):
        res *= i
    return res


class Binomial():
    """ Class representing a Binomial Distribution """

    def __init__(self, data=None, n=1, p=0.5):
        """ Custom init method """

        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            elif p <= 0 or p >= 1:
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

    def pmf(self, k):
        """ Calculates PMF for given k """
        k = int(k)
        if k < 0:
            return 0

        n_choose_k = factorial(self.n) / (factorial(k) * factorial(self.n - k))
        pmf = n_choose_k * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pmf
