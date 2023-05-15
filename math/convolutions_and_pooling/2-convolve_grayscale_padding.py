#!/usr/bin/env python3
""" Task 2. Convolution with padding """


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images
    with custom padding:

    @images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    @kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    @padding is a tuple of (ph, pw)
    ph is the padding for the height of the image
    pw is the padding for the width of the image
    Only two for loops allowed

    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    convoluted = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            image = padded[:, i + ph: i + ph + kh, j + pw: j + pw + kw]
            convoluted[:, i, j] = np.sum(image * kernel, axis=(1, 2))
    return convoluted
