#!/usr/bin/env python3
# _*_coding:utf-8 _*_

import os
import numpy as np
import tensorflow as tf
import random
import time

"""
读取txt中的数据，并将数据保存成tfrecord文件
arg:
    txt_filename: 是txt保存的路径+文件名 'data/'
    tfrecord_path：tfrecord文件将要保存的路径及名称 'data/'
"""


def txt_to_tfrecord(txt_filename, tfrecord_path):
    # 第一步：生成TFRecord Writer
    writer = tf.python_io.TFRecordWriter(tfrecord_path)

    # 第二步：读取TXT数据，并分割出样本数据和标签
    file = open(txt_filename)
    for data_line in file.readlines():  # 每一行
        data_line = data_line.strip('\n')  # 去掉换行符
        sample = []
        spls = data_line.split('/', 1)[0]  # 样本
        for m in spls.split(' '):
            sample.append(int(m))
        label = data_line.split('/', 1)[1]  # 标签
        label = int(label)
        print('sample:', sample, 'labels:', label)

        # 第三步： 建立feature字典，tf.train.Feature()对单一数据编码成feature
        feature = {'sample': tf.train.Feature(int64_list=tf.train.Int64List(value=sample)),
                   'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}
        # 第四步：可以理解为将内层多个feature的字典数据再编码，集成为features
        features = tf.train.Features(feature=feature)
        # 第五步：将features数据封装成特定的协议格式
        example = tf.train.Example(features=features)
        # 第六步：将example数据序列化为字符串
        Serialized = example.SerializeToString()
        # 第七步：将序列化的字符串数据写入协议缓冲区
        writer.write(Serialized)
    # 记得关闭writer和open file的操作
    writer.close()
    file.close()
    return
