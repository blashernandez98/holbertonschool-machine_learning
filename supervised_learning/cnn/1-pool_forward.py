#!/usr/bin/env python3
""" Task 1. Pooling Forward Prop """


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network:

    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw)
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg
    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ch = int(((h_prev - kh) / sh) + 1)
    cw = int(((w_prev - kw) / sw) + 1)
    pooled = np.zeros((m, ch, cw, c_prev))

    for i in range(ch):
        for j in range(cw):
            si = i * sh
            sj = j * sw
            image = A_prev[:, si: si + kh, sj: sj + kw, :]
            if mode == 'max':
                result = np.max(image, axis=(1, 2))
            elif mode == 'avg':
                result = np.mean(image, axis=(1, 2))
            pooled[:, i, j, :] = result

    return pooled
