#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/10 07:01 PM
# @Author  : zhangzhen
# @Site    :
# @File    : keras2_demo.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

print(tf.__version__)
print(tf.keras.__version__)

class MyModel(tf.keras.Model):
    """docstring for MyModel"tf.keras.Model """
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        self.layers1 = layers.Dense(32, activation='relu')
        self.layers2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        h1 = self.layers1(inputs)
        out = self.layers2(h1)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

class MyLayer(layers.Layer):
    """docstring for MyLayer"""
    def __init__(self, output_dim, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        self.kernel = self.add_weight(name='kernel1', shape=shape, initializer='uniform', trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def model(train_x, train_y):
    model = MyModel(num_classes=10)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=16, epochs=50)


def layer(train_x, train_y, val_x, val_y):
    model = tf.keras.Sequential([MyLayer(10), layers.Activation('softmax')])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'), tf.keras.callbacks.TensorBoard(log_dir='./logs')]
    model.fit(train_x, train_y, batch_size=16, epochs=50, callbacks=callbacks, validation_data=(val_x, val_y))


train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))
val_x = np.random.random((200, 72))
val_y = np.random.random((200, 10))
# model(train_x, train_y)
layer(train_x, train_y, val_x, val_y)