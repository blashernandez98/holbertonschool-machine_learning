#!/usr/bin/env python3
""" Task 0. Convolutional Forward Prop """


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a
    convolutional layer of a neural network:
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
        kh is the filter height
        kw is the filter width
        c_prev is the number of channels in the previous layer
        c_new is the number of channels in the output
    b is a numpy.ndarray of shape (1, 1, 1, c_new)
    activation is an activation function applied to the convolution
    padding is a string that is either same or valid
    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
    you may import numpy as np
    Returns: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = int((((h_prev - 1) * sh + kh - h_prev) / 2) + 1)
        pw = int((((w_prev - 1) * sw + kw - w_prev) / 2) + 1)
    elif padding == 'valid':
        ph, pw = 0, 0
    padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')
    ch = int(((h_prev + (2 * ph) - kh) / sh) + 1)
    cw = int(((w_prev + (2 * pw) - kw) / sw) + 1)
    convoluted = np.zeros((m, ch, cw, c_new))

    for ker in range(c_new):
        curr_kernel = W[:, :, :, ker]
        for i in range(ch):
            for j in range(cw):
                si = i * sh
                sj = j * sw
                image = padded[:, si: si + kh, sj: sj + kw, :]
                result = image * curr_kernel
                result = result.sum(axis=1)
                result = result.sum(axis=1)
                result = result.sum(axis=1)
                result = result + b[:, :, :, ker]
                result = activation(result)
                convoluted[:, i, j, ker] = result

    return convoluted
