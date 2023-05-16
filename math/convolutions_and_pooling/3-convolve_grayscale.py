#!/usr/bin/env python3
""" Task 3. Strided Convolution """


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images:

    images is a numpy.ndarray with shape (m, h, w)
    containing multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw)
    containing the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel
    padding is either a tuple of (ph, pw), 'same', or 'valid'
        if 'same', performs a same convolution
        if 'valid', performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
        The image should be padded with 0's
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    You are only allowed to use two for loops;
    Returns: a numpy.ndarray containing the convolved images

    """

    m, h, w = images.shape
    kh, kw = kernel.shape
    if padding == 'same':
        ph = max((kh - 1) // 2, kh // 2)
        pw = max((kw - 1) // 2, kw // 2)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    sh, sw = stride
    ch = (h + 2 * ph - kh) // sh + 1
    cw = (w + 2 * pw - kw) // sw + 1
    convoluted = np.zeros((m, ch, cw))
    for i in range(ch):
        for j in range(cw):
            si = i * sh
            sj = j * sw
            image = padded[:, si: si + kh, sj: sj + kw]
            convoluted[:, i, j] = np.sum(image * kernel, axis=(1, 2))
    return convoluted
