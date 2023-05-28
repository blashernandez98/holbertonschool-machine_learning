#!/usr/bin/env python3
""" Task 7. DenseNet121 """


import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in:
    Densely Connected Convolutional Networks (2018)
    (https://arxiv.org/pdf/1608.06993.pdf)

    - growth_rate is the growth rate
    - compression is the compression factor
    - You can assume the input data will have shape (224, 224, 3)
    - All convolutions should be preceded by Batch Normalization
        and a rectified linear activation (ReLU), respectively
    - All weights should use he normal initialization
    - You may use:
        dense_block = __import__('5-dense_block').dense_block
        transition_layer = __import__('6-transition_layer').transition_layer
    Returns: the keras model
    """
    init = K.initializers.he_normal()
    inputs = K.Input(shape=(224, 224, 3))
    batch_norm = K.layers.BatchNormalization()(inputs)
    activation = K.layers.Activation('relu')(batch_norm)
    conv = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                           strides=(2, 2), padding='same',
                           kernel_initializer=init)(activation)
    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                     padding='same')(conv)
    nb_filters = 2 * growth_rate
    X, nb_filters = dense_block(max_pool, nb_filters, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                         strides=(1, 1))(X)
    dense = K.layers.Dense(units=1000, activation='softmax',
                           kernel_initializer=init)(avg_pool)

    model = K.models.Model(inputs=X, outputs=dense)

    return model
