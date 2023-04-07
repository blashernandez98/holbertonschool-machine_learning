#!/usr/bin/env python3
""" Normal distribution module """


class Normal():
    """ Class representing a normal distribution """

    def __init__(self, data=None, mean=0., stddev=1.):
        """ Custom init method """

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = mean
            self.stddev = stddev
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data) / len(data)
            stddev = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = stddev ** 0.5

    def z_score(self, x):
        """ Calculates the z-score of a given value """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculates the value for a given z-score """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """ Calculates PDF for given x value """
        pi = 3.1415926536
        e = 2.7182818285
        first_term = 1 / (self.stddev * (pi * 2) ** 0.5)
        expo = (((x - self.mean) / self.stddev) ** 2) * -0.5

        return (first_term * (e ** expo))
