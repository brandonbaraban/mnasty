#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


def main():
    x_train, y_train, x_test, y_test = load_data()
    model = get_model()
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)


def load_data():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print(x_train.shape)
    return x_train, y_train, x_test, y_test


def get_model():
    # simple one hidden layer with dropout
    model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    main()
