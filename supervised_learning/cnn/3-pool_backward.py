#!/usr/bin/env python3
""" Task 3. Pooling Back Prop """


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network:

    dA is a numpy.ndarray of shape (m, h_new, w_new, c_new)
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c is the number of channels
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c)
    containing the output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
    kernel_shape is a tuple of (kh, kw)
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg
    Returns: the partial derivatives with respect to the previous layer
    """

    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    if mode == 'max':
                        A_prev_slice = A_prev[i, h * sh: h * sh + kh,
                                              w * sw: w * sw + kw, c]
                        mask = (A_prev_slice == np.max(A_prev_slice))
                        dA_prev[i, h * sh: h * sh + kh,
                                w * sw: w * sw + kw, c] +=\
                            dA[i, h, w, c] * mask
                    elif mode == 'avg':
                        dA_prev[i, h * sh: h * sh + kh,
                                w * sw: w * sw + kw, c] +=\
                            dA[i, h, w, c] / (kh * kw)

    return dA_prev
