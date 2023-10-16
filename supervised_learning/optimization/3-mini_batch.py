#!/usr/bin/env python3
""" 3. Mini-batch """

import tensorflow as tf
import numpy as np

shuffle_data = __import__("2-shuffle_data").shuffle_data


def train_mini_batch(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    batch_size=32,
    epochs=5,
    load_path="/tmp/model.ckpt",
    save_path="/tmp/model.ckpt",
):
    """
    Function that trains a loaded neural network model
    using mini-batch gradient descent
    """

    # Start a TensorFlow session to run the training operations
    with tf.Session() as sess:
        # Load the model graph and restore the session
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        # Access to the placeholders of the graph
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        train_op = tf.get_collection("train_op")[0]
        loss = tf.get_collection("loss")[0]
        m = X_train.shape[0]

        # For each epoch
        for i in range(epochs):

            train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})

            valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            print(
                "After {} epochs:".format(i),
                "\n\tTraining Cost: {}".format(train_cost),
                "\n\tTraining Accuracy: {}".format(train_accuracy),
                "\n\tValidation Cost: {}".format(valid_cost),
                "\n\tValidation Accuracy: {}".format(valid_accuracy),
            )

            # Shuffle the data in the training
            X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

            # Mini-batch
            if m % batch_size == 0:
                complete = m // batch_size
            else:
                complete = m // batch_size + 1

            for j in range(complete):
                start = j * batch_size
                end = (j + 1) * batch_size
                if end > m:
                    end = m
                X_mini = X_shuffled[start:end]
                Y_mini = Y_shuffled[start:end]

                sess.run(train_op, feed_dict={x: X_mini, y: Y_mini})

                if j != 0 and (j + 1) % 100 == 0:
                    mini_cost = sess.run(loss, feed_dict={x: X_mini, y: Y_mini})
                    mini_accuracy = sess.run(
                        accuracy, feed_dict={x: X_mini, y: Y_mini}
                    )
                    print(
                        "\tStep {}:".format(j + 1),
                        "\n\t\tCost: {}".format(mini_cost),
                        "\n\t\tAccuracy: {}".format(mini_accuracy),
                    )

        # Save the model after training
        save_path = saver.save(sess, save_path)
    return save_path

