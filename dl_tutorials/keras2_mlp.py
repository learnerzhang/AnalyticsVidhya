#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/17 07:01 PM
# @Author  : zhangzhen
# @Site    :
# @File    : keras2_demo.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np


def build(input_dim=20, out_dim=10, activation='softmax'):
    model = tf.keras.Sequential()
    model.add(layers.Dense(64,activation='relu', input_dim=input_dim))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(out_dim, activation=activation))
    return model

def softmax():
    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)


    model = build()
    sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    keras.utils.plot_model(model, 'sequential_mlp.png', show_shapes=True)

    model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)

def binary():
    x_train = np.random.random((1000, 20))
    y_train = np.random.randint(2, size=(1000, 1))
    x_test = np.random.random((100, 20))
    y_test = np.random.randint(2, size=(100, 1))
    model = build(input_dim=20, out_dim=1, activation='sigmoid')

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    keras.utils.plot_model(model, 'sequential_mlp.png', show_shapes=True)

    model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)

if __name__ == '__main__':
    # softmax()
    binary()
    