#!/usr/bin/env python3
""" Task 6 module """


import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    # Forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)
    # Loss function
    loss = calculate_loss(y, y_pred)
    # Optimization
    train_op = create_train_op(loss, alpha)
    # Accuracy operations
    accuracy = calculate_accuracy(y, y_pred)
    accuracy_valid = calculate_accuracy(
                     Y_valid, forward_prop(X_valid, layer_sizes, activations))
    # Initialize variables and create saver
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        tf.add_to_collection('X', x)
        tf.add_to_collection('Y', y)
        tf.add_to_collection('Y_pred', y_pred)
        tf.add_to_collection('loss', loss)
        tf.add_to_collection('accuracy', accuracy)
        tf.add_to_collection('train_op', train_op)

        # Train the model:
        for i in range(iterations + 1):
            cost_train, acc_train = sess.run(
                [loss, accuracy], feed_dict={X: X_train, Y: Y_train})
            cost_valid, acc_valid = sess.run(
                [loss, accuracy], feed_dict={X: X_valid, Y: Y_valid})

            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_train))
                print("\tTraining Accuracy: {}".format(acc_train))
                print("\tValidation Cost: {}".format(cost_valid))
                print("\tValidation Accuracy: {}".format(acc_valid))

            if i < iterations:
                sess.run(train_op, feed_dict={X: X_train, Y: Y_train})

        # Save the trained model using a TensorFlow Saver object:
        save_path = saver.save(sess, save_path)

    return save_path
