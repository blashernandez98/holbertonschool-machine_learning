#!/usr/bin/env python3
""" Task 5. LeNet-5 (Keras) """


import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of the LeNet-5 architecture using keras:
    X is a K.Input of shape (m, 28, 28, 1)
    containing the input images for the network
        m is the number of images
    The model should consist of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels
    with the he_normal initialization method
    All hidden layers requiring activation should
    use the relu activation function
    you may import tensorflow.keras as K
    Returns: a K.Model compiled to use Adam optimization
    (with default hyperparameters) and accuracy metrics
    """
    initializer = K.initializers.he_normal()

    conv1 = K.layers.Conv2D(6, kernel_size=(5, 5),
                            padding='same', activation='relu',
                            kernel_initializer=initializer)(X)

    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)

    conv2 = K.layers.Conv2D(16, kernel_size=(5, 5),
                            padding='valid', activation='relu',
                            kernel_initializer=initializer)(pool1)

    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    flatten = K.layers.Flatten()(pool2)

    fc1 = K.layers.Dense(120, activation='relu',
                         kernel_initializer=initializer)(flatten)

    fc2 = K.layers.Dense(84, activation='relu',
                         kernel_initializer=initializer)(fc1)

    output = K.layers.Dense(10, kernel_initializer=initializer)(fc2)

    softmax_out = K.layers.Softmax()(output)

    model = K.models.Model(inputs=X, outputs=softmax_out)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model
