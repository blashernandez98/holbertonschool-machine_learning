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

    def pmf(self, k):
        """ Returns PMF of @k number of 'successes' """
        e = 2.7182818285
        if k is None or k < 0:
            return 0
        k = int(k)
        fact = 1
        for i in range(1, k+1):
            fact *= i
        return (self.lambtha ** k * (e ** (-self.lambtha)) / fact)

    def cdf(self, k):
        """ Returns CDF of @k number of 'successes' """
        e = 2.7182818285
        k = int(k)
        cdf = 0
        for n in range(0, k+1):
            fact = 1
            for i in range(1, n+1):
                fact *= i
            cdf += self.lambtha ** n * (e ** (-self.lambtha)) / fact
        return cdf
