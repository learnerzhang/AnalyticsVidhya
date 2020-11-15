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
import numpy as np

print(tf.__version__)
print(tf.keras.__version__)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

def build_sample_net():
    inputs = tf.keras.Input(shape=(784,), name='img')
    h1 = layers.Dense(32, activation='relu')(inputs)
    h2 = layers.Dense(32, activation='relu')(h1)
    outputs = layers.Dense(10, activation='softmax')(h2)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='mnist model')

    model.summary()
    keras.utils.plot_model(model, 'mnist_model.png')
    keras.utils.plot_model(model, 'model_info.png', show_shapes=True)


    x_train = x_train.reshape(-1, 28*28).astype('float32') /255
    x_test = x_test.reshape(-1, 28*28).astype('float32') /255
    print(x_train.shape, y_train.shape)

    model.compile(optimizer=keras.optimizers.RMSprop(),
                 loss='sparse_categorical_crossentropy', # 直接填api，后面会报错
                 metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)

    test_scores = model.evaluate(x_test, y_test, verbose=0)
    print('test loss:', test_scores[0])
    print('test acc:', test_scores[1])


    model.save('model_save.h5')
    del model
    model = keras.models.load_model('model_save.h5')


def build_shared_net():
    encode_input = keras.Input(shape=(28,28,1), name='img')
    h1 = layers.Conv2D(16, 3, activation='relu')(encode_input)
    h1 = layers.Conv2D(32, 3, activation='relu')(h1)
    h1 = layers.MaxPool2D(3)(h1)
    h1 = layers.Conv2D(32, 3, activation='relu')(h1)
    h1 = layers.Conv2D(16, 3, activation='relu')(h1)
    encode_output = layers.GlobalMaxPool2D()(h1)
    encode_model = keras.Model(inputs=encode_input, outputs=encode_output, name='encoder')
    encode_model.summary()

    h2 = layers.Reshape((4, 4, 1))(encode_output)
    h2 = layers.Conv2DTranspose(16, 3, activation='relu')(h2)
    h2 = layers.Conv2DTranspose(32, 3, activation='relu')(h2)
    h2 = layers.UpSampling2D(3)(h2)
    h2 = layers.Conv2DTranspose(16, 3, activation='relu')(h2)
    decode_output = layers.Conv2DTranspose(1, 3, activation='relu')(h2)

    autoencoder = keras.Model(inputs=encode_input, outputs=decode_output, name='autoencoder')
    autoencoder.summary()


def build_mulit_input_output_net():
    num_words = 2000
    num_tags = 12
    num_departments = 4

    # 输入
    body_input = keras.Input(shape=(None,), name='body')
    title_input = keras.Input(shape=(None,), name='title')
    tag_input = keras.Input(shape=(num_tags,), name='tag')

    # 嵌入层
    body_feat = layers.Embedding(num_words, 64)(body_input)
    title_feat = layers.Embedding(num_words, 64)(title_input)

    # 特征提取层
    body_feat = layers.LSTM(32)(body_feat)
    title_feat = layers.LSTM(128)(title_feat)
    features = layers.concatenate([title_feat, body_feat, tag_input])

    # 分类层
    priority_pred = layers.Dense(1, activation='sigmoid', name='priority')(features)
    department_pred = layers.Dense(num_departments, activation='softmax', name='department')(features)

    # 构建模型
    model = keras.Model(inputs=[body_input, title_input, tag_input], outputs=[priority_pred, department_pred])
    model.summary()
    keras.utils.plot_model(model, 'multi_model.png', show_shapes=True)
    model.compile(optimizer=keras.optimizers.RMSprop(1e-3),loss={'priority': 'binary_crossentropy','department': 'categorical_crossentropy'},loss_weights=[1., 0.2])

    # 载入输入数据
    title_data = np.random.randint(num_words, size=(1280, 10))
    body_data = np.random.randint(num_words, size=(1280, 100))
    tag_data = np.random.randint(2, size=(1280, num_tags)).astype('float32')

    # 标签
    priority_label = np.random.random(size=(1280, 1))
    department_label = np.random.randint(2, size=(1280, num_departments))

    # 训练
    history = model.fit(
        {'title': title_data, 'body':body_data, 'tag':tag_data},
        {'priority':priority_label, 'department':department_label},
        batch_size=32,
        epochs=5
    )

def build_res_net():
    inputs = keras.Input(shape=(32,32,3), name='img')
    h1 = layers.Conv2D(32, 3, activation='relu')(inputs)
    h1 = layers.Conv2D(64, 3, activation='relu')(h1)
    block1_out = layers.MaxPooling2D(3)(h1)

    h2 = layers.Conv2D(64, 3, activation='relu', padding='same')(block1_out)
    h2 = layers.Conv2D(64, 3, activation='relu', padding='same')(h2)
    block2_out = layers.add([h2, block1_out])

    h3 = layers.Conv2D(64, 3, activation='relu', padding='same')(block2_out)
    h3 = layers.Conv2D(64, 3, activation='relu', padding='same')(h3)
    block3_out = layers.add([h3, block2_out])

    h4 = layers.Conv2D(64, 3, activation='relu')(block3_out)
    h4 = layers.GlobalMaxPool2D()(h4)
    h4 = layers.Dense(256, activation='relu')(h4)
    h4 = layers.Dropout(0.5)(h4)
    outputs = layers.Dense(10, activation='softmax')(h4)

    model = keras.Model(inputs, outputs, name='small resnet')
    model.summary()
    keras.utils.plot_model(model, 'small_resnet_model.png', show_shapes=True)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = y_train.astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model.compile(optimizer=keras.optimizers.RMSprop(1e-3), loss='categorical_crossentropy', metrics=['acc'])
    model.fit(x_train, y_train, batch_size=64, epochs=1, validation_split=0.2)

if __name__ == '__main__':
    # build_sample_net()
    # build_shared_net()
    # build_mulit_input_output_net()
    # build_res_net()
    build_res_net()
