#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf


def main():
    x_train, y_train, x_test, y_test = load_data(normalize=True, pca_k=None)
    model = get_model(x_train[0].shape)
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)


def load_data(normalize=False, pca_k=None):
    '''
    loads mnist data with desired transormations applied.
    normalize (optional bool): if true, normalizes data
    pca (optional int): expected in range [1, 784].
        if present, performs principal component analysis with rank {pca_k} result.
    '''
    mnist = tf.keras.datasets.mnist # each matrix of data is shape (samples, variables)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    if normalize:
        x_train, x_test = x_train / 255.0, x_test / 255.0
    if pca_k is not None:
        x_train = pca(x_train, pca_k)
    return x_train, y_train, x_test, y_test


def pca(a, k):
    a -= np.mean(a, axis=0) # zero-out sample mean
    print('calculating svd...')
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    print('done.')
    uk = u[:, :k]
    sk = s[:k]
    vhk = vh[:k, :]
    return uk @ np.diag(sk) @ vhk


def get_model(input_shape, hidden_units=512, dropout_rate=0.2):
    # simple one hidden layer with dropout
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(hidden_units, input_shape=input_shape, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    main()
