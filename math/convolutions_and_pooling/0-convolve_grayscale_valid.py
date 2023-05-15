#!/usr/bin/env python3
""" Task 0. Valid Convolution """


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images:

    @images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    @kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    You are only allowed to use two for loops;
    any other loops of any kind are not allowed
    Returns: a numpy.ndarray containing the convolved images
    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    convoluted = np.zeros((m, h - kh + 1, w - kw + 1))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            image = images[:, i: i + kh, j: j + kw]
            convoluted[:, i, j] = np.sum(image * kernel, axis=(1, 2))
    return convoluted
