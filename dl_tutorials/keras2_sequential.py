#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 07:01 PM
# @Author  : zhangzhen
# @Site    :
# @File    : keras2_demo.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_seq1_model():
    model = tf.keras.Sequential([
        layers.Dense(32, input_shape=(100,)),
        layers.Activation('relu'),
        layers.Dense(10),
        layers.Activation('softmax'),
    ])
    return model

def get_seq2_model():
    """
    docstring
    """
    model = tf.keras.Sequential()
    model.add(layers.Dense(32, activation='relu', input_dim=100))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def build():
    model = get_seq2_model()
    model.summary()
    keras.utils.plot_model(model, 'sequential_model.png', show_shapes=True)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    import numpy as np
    data = np.random.random((1000, 100))
    labels = np.random.randint(2, size=(1000, 1))

    model = build()
    # 训练模型，以 32 个样本为一个 batch 进行迭代
    model.fit(data, labels, epochs=10, batch_size=32)
    