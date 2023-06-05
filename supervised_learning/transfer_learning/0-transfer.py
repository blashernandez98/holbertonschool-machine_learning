from turtle import mode
import tensorflow as tf
from tensorflow import keras as K
import os


def preprocess_data(X, Y):
    """
    Pre-processes the data for the model:
    - X is a numpy.ndarray of shape (m, 32, 32, 3)
        containing the CIFAR 10 data, where m is the number of data points
    - Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.inception_v3.preprocess_input(X)

    Y_p = K.utils.to_categorical(Y, num_classes=10)

    return X_p, Y_p


if __name__ == "__main__":
    tf.random.set_seed(seed=43)
    model_path = "cifar10v5.h5"
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_valid, Y_valid = preprocess_data(X_valid, Y_valid)

    if os.path.exists(model_path):
        model = K.models.load_model(model_path)
        print("Loaded model from disk")
    else:
        # Load InceptionV3 base model
        base_model = K.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
        )

        # Freeze base model
        base_model.trainable = False
        # Create input layer
        inputs = K.Input(shape=(32, 32, 3))
        # Create data augmentation layers
        aug1 = K.layers.experimental.preprocessing.RandomFlip("horizontal")(inputs)
        aug2 = K.layers.experimental.preprocessing.RandomRotation(0.1)(aug1)
        aug3 = K.layers.experimental.preprocessing.RandomZoom(0.2)(aug2)
        # Resize images to match InceptionV3 input dimensions
        resize = K.layers.Lambda(lambda image: tf.image.resize(image, (150, 150)))(aug3)
        # Pass resized images to base model
        inception = base_model(resize, training=False)
        # Pooling layer
        pool = K.layers.GlobalAveragePooling2D()(inception)
        # Dropout layer for normalization
        dropout = K.layers.Dropout(0.5)(pool)
        # Flatten and 2 fully connected layers with batch normalization
        flat = K.layers.Flatten()(dropout)
        dense = K.layers.Dense(256, activation="relu")(flat)
        batch = K.layers.BatchNormalization()(dense)
        dropout2 = K.layers.Dropout(0.5)(batch)
        dense2 = K.layers.Dense(128, activation="relu")(dropout2)
        batch2 = K.layers.BatchNormalization()(dense2)
        # Dropout layer for normalization
        dropout3 = K.layers.Dropout(0.5)(batch2)
        # Output layer
        outputs = K.layers.Dense(10, activation="softmax")(dropout3)
        # Create model
        model = K.Model(inputs, outputs)

    model.summary()

    model.compile(
        optimizer=K.optimizers.RMSprop(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Add callback checkpoint
    callbacks = []

    callbacks.append(
        K.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        )
    )

    early_stopping = K.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=1,
        mode="max",
    )

    callbacks.append(early_stopping)

    history = model.fit(
        X_train,
        Y_train,
        batch_size=32,
        epochs=5,
        validation_data=(X_valid, Y_valid),
        callbacks=callbacks,
    )

    model.save(model_path)
