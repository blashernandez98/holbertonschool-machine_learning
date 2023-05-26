#!/usr/bin/env python3
""" Task 1. Inception Network """


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network as described in
    Going Deeper with Convolutions (2014) https://arxiv.org/pdf/1409.4842.pdf

    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the inception block
    should use a rectified linear activation (ReLU)
    Returns: the keras model
    """

    init = K.initializers.he_normal()
    # input data variable
    input_data = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(
        filters=64, kernel_size=(7, 7), strides=(2, 2),
        padding='same', kernel_initializer=init,
        activation='relu')(input_data)

    pool1 = K.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(2, 2), padding='same')(conv1)

    conv2R = K.layers.Conv2D(
        filters=64, kernel_size=(1, 1), strides=(1, 1),
        padding='same', kernel_initializer=init, activation='relu')(pool1)
    conv2 = K.layers.Conv2D(
        filters=192, kernel_size=(3, 3), strides=(1, 1),
        padding='same', kernel_initializer=init, activation='relu')(conv2R)

    pool2 = K.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(2, 2), padding='same')(conv2)

    inception3a = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    inception3b = inception_block(inception3a, [128, 128, 192, 32, 96, 64])
    pool3 = K.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(2, 2), padding='same')(inception3b)

    inception4a = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    inception4b = inception_block(inception4a, [160, 112, 224, 24, 64, 64])
    inception4c = inception_block(inception4b, [128, 128, 256, 24, 64, 64])
    inception4d = inception_block(inception4c, [112, 144, 288, 32, 64, 64])
    inception4e = inception_block(inception4d, [256, 160, 320, 32, 128, 128])
    pool4 = K.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(2, 2), padding='same')(inception4e)
    inception5a = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    inception5b = inception_block(inception5a, [384, 192, 384, 48, 128, 128])
    pool5 = K.layers.AveragePooling2D(
        pool_size=(7, 7), strides=(1, 1))(inception5b)

    drop = K.layers.Dropout(rate=0.4)(pool5)
    softmax = K.layers.Dense(units=1000, activation='softmax',
                             kernel_initializer=init)(drop)

    model = K.models.Model(inputs=input_data, outputs=softmax)

    return model
