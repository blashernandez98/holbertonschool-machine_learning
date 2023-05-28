#!/usr/bin/env python3
""" Task 4. ResNet-50 """


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in:
    Deep Residual Learning for Image Recognition (2015)
    (https://arxiv.org/pdf/1512.03385.pdf)

    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should be
    followed by batch normalization along the channels axis
    and a rectified linear activation (ReLU), respectively.
    All weights should use he normal initialization
    You may use:
        identity_block = __import__('2-identity_block').identity_block
        projection_block = __import__('3-projection_block').projection_block
    Returns: the keras model
    """

    init = K.initializers.he_normal()

    X = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(
        filters=64, kernel_size=(7, 7),
        padding='same', strides=(2, 2),
        kernel_initializer=init)(X)

    batch1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation('relu')(batch1)

    pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2),
        padding='same')(relu1)

    conv2_1 = projection_block(pool1, [64, 64, 256], 1)
    conv2_2 = identity_block(conv2_1, [64, 64, 256])
    conv2_3 = identity_block(conv2_2, [64, 64, 256])

    conv3_1 = projection_block(conv2_3, [128, 128, 512])
    conv3_2 = identity_block(conv3_1, [128, 128, 512])
    conv3_3 = identity_block(conv3_2, [128, 128, 512])
    conv3_4 = identity_block(conv3_3, [128, 128, 512])

    conv4_1 = projection_block(conv3_4, [256, 256, 1024])
    conv4_2 = identity_block(conv4_1, [256, 256, 1024])
    conv4_3 = identity_block(conv4_2, [256, 256, 1024])
    conv4_4 = identity_block(conv4_3, [256, 256, 1024])
    conv4_5 = identity_block(conv4_4, [256, 256, 1024])
    conv4_6 = identity_block(conv4_5, [256, 256, 1024])

    conv5_1 = projection_block(conv4_6, [512, 512, 2048])
    conv5_2 = identity_block(conv5_1, [512, 512, 2048])
    conv5_3 = identity_block(conv5_2, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7), strides=(1, 1))(conv5_3)

    output = K.layers.Dense(
        units=1000, activation='softmax',
        kernel_initializer=init)(avg_pool)

    model = K.models.Model(inputs=X, outputs=output)

    return model
