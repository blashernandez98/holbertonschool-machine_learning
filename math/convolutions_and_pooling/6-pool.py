#!/usr/bin/env python3
""" Task 6. Pooling """


import numpy as np


def pool(images, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs pooling on images:

    images is a numpy.ndarray with shape (m, h, w, c)
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images
        c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw)
        kh is the height of the kernel
        kw is the width of the kernel
    stride is a tuple of (sh, sw)
        sh is the stride for the height of the image
        sw is the stride for the width of the image
    mode indicates the type of pooling
        max indicates max pooling
        avg indicates average pooling
    You are only allowed to use two for loops
    Returns: a numpy.ndarray containing the pooled images
    """

    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ph = int(((h - kh) / sh) + 1)
    pw = int(((w - kw) / sw) + 1)
    pooled = np.zeros((m, ph, pw, c))

    for channel in range(c):
        for i in range(ph):
            for j in range(pw):
                si = i * sh
                sj = j * sw
                image = images[:, si: si + kh, sj: sj + kw, :]
                if mode == 'max':
                    result = np.max(image, axis=(1, 2))
                elif mode == 'avg':
                    result = np.mean(image, axis=(1, 2))
                pooled[:, i, j, :] = result

    return pooled
