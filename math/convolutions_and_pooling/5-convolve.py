#!/usr/bin/env python3
""" Task 5. Multiple Kernels """


import numpy as np


def convolve(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels:

    images is a numpy.ndarray with shape (m, h, w, c)
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel is a numpy.ndarray with shape (kh, kw, c)
        kh is the height of the kernel
        kw is the width of the kernel
    padding is either a tuple of (ph, pw), 'same', or 'valid'
        if 'same', performs a same convolution
        if 'valid', performs a valid convolution
        if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
        the image should be padded with 0's
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    Returns: a numpy.ndarray containing the convolved images
    """

    m, h, w, c = images.shape
    kh, kw, _, kc = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding
    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    ch = int(((h + (2 * ph) - kh) / sh) + 1)
    cw = int(((w + (2 * pw) - kw) / sw) + 1)
    convoluted = np.zeros((m, ch, cw, kc))

    for ker in range(kc):
        curr_kernel = kernel[:, :, :, ker]
        for i in range(ch):
            for j in range(cw):
                si = i * sh
                sj = j * sw
                image = padded[:, si: si + kh, sj: sj + kw, :]
                result = image * curr_kernel
                result = result.sum(axis=1)
                result = result.sum(axis=1)
                result = result.sum(axis=1)
                convoluted[:, i, j, ker] = result

    return convoluted
