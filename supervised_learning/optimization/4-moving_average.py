#!/usr/env python3
""" 3. Moving Average """


def moving_average(data, beta):
    """
    function that calculates the weighted moving average of a data set
    using bias correction
    """

    v = 0
    avg = []
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        avg.append(v / (1 - beta ** (i + 1)))
    return avg
