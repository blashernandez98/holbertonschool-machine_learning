#!/usr/bin/env python3

def create_placeholders(nx, classes):
    import tensorflow as tf

    x = tf.placeholder(tf.float32, shape=(None, nx))
    y = tf.placeholder(tf.float32, shape=(None, classes))
    return (x, y)
