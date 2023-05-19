#!/usr/bin/env python3
""" Task 2. Convolutional Back Propagation """


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network:
    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new)
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
        kh is the filter height
        kw is the filter width
    b is a numpy.ndarray of shape (1, 1, 1, c_new)
    padding is a string that is either same or valid
    stride is a tuple of (sh, sw)
        sh is the stride for the height
        sw is the stride for the width
    Returns: the partial derivatives with respect to the previous layer
    (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0

    padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    dA_prev = np.zeros(padded.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    dZ_cut = dZ[i, h, w, c]
                    dA_prev[i, h * sh: h * sh + kh, w * sw: w * sw + kw, :]\
                        += (dZ_cut * W[:, :, :, c])
                    dW[:, :, :, c] += padded[i, h * sh: h * sh + kh,
                                             w * sw: w * sw + kw, :] * dZ_cut
                    db[:, :, :, c] += dZ_cut

    dA_prev = dA_prev[:, ph: dA_prev.shape[1] -
                      ph, pw: dA_prev.shape[2] - pw, :]

    return dA_prev, dW, db
