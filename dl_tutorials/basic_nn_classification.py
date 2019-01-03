#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/20 10:45 AM
# @Author  : zhangzhen
# @Site    : 
# @File    : basic_nn_classification.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


def get_data():
    # data = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')
    data = input_data.read_data_sets('data/fashion', one_hot=True, reshape=False)
    # data.train.next_batch(1000)
    (train_images, train_labels), (test_images, test_labels) = (data.train.images, data.train.labels), (
        data.test.images, data.test.labels)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print(train_images.shape, train_labels.shape)
    print(test_images.shape, test_labels.shape)

    return train_images, train_labels, test_images, test_labels, class_names


def show_image(img):
    if img.ndim == 3:
        img = img[:, :, 0]
    plt.subplot(221)
    plt.imshow(img)
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.subplot(223)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.colorbar()
    plt.subplot(224)
    plt.imshow(img, cmap=plt.cm.gray_r)
    plt.colorbar()
    while True:
        try:
            plt.show()
        except UnicodeDecodeError:
            continue
        break


def display(train_images, train_labels, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        tmp = train_images[i].reshape(28, -1)
        plt.imshow(tmp)  # cmap = plt.cm.binary
        # print(np.where(train_labels[i]==1)[0][0])
        label_idx = np.where(train_labels[i] == 1)[0][0]
        plt.xlabel(class_names[label_idx])
    while True:
        try:
            plt.show()
        except UnicodeDecodeError:
            continue
        break


class NN:

    def __init__(self):
        pass

    def build(self):
        # 构建模型
        pass

    def train(self):
        """训练模型"""
        pass
    def predict(self):
        """预测"""
        pass


if __name__ == '__main__':
    # get data
    train_images, train_labels, test_images, test_labels, class_names = get_data()
    # show_image(train_images[0])
    # display(train_images, train_labels, class_names)
