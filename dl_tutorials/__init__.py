#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/21 2:30 PM
# @Author  : zhangzhen
# @Site    : 
# @File    : __init__.py
# @Software: PyCharm
import tensorflow as tf
from numpy.random import RandomState as rdm
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

class MLP:

    def __init__(self, in_size=10, out_size=2, hiddens=[10], act_function=None) -> object:
        self.x_dimension = in_size
        self.y_dimension = out_size
        self.build(in_size, out_size, hiddens=hiddens, act_function=act_function)


    def build(self, in_size, out_size, hiddens=[], act_function=tf.nn.relu):

        def add_layer(inputs: object, in_size: object, out_size: object, act_function: object = None) -> object:
            W = tf.Variable(tf.random_normal([in_size, out_size]))
            b = tf.Variable(tf.constant(0.1, shape=[out_size]))
            Wx_plus_b = tf.matmul(inputs, W) + b
            if act_function:
                outputs = act_function(Wx_plus_b)
            else:
                outputs = Wx_plus_b
            logging.info("tmp hidden layer out: {}".format(outputs))
            return outputs

        self.x = tf.placeholder(dtype=tf.float32, shape=(None, in_size), name='X-input')
        self.y_ = tf.placeholder(dtype=tf.float32, shape=(None, out_size), name='y-input')

        tmp_in_size = in_size
        tmp_inputs = self.x
        for hidden in hiddens:
            tmp_outputs = add_layer(tmp_inputs, tmp_in_size, hidden, act_function=act_function)
            tmp_in_size = hidden
            tmp_inputs = tmp_outputs
        self.y = add_layer(tmp_inputs, tmp_in_size, out_size, act_function=None)
        logging.info("last out: {}".format(self.y))

        self.cross_entropy = -tf.reduce_mean(self.y_ * tf.log(tf.clip_by_value(self.y, 1e-10, 1.0)))
        self.step = tf.train.AdamOptimizer(0.001).minimize(self.cross_entropy)
        logging.info("loss: {}".format(self.cross_entropy))

    def train(self, steps=5000, batch_size=8):

        X, Y = self.generate_data(size=128)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # logging.info()
            for i in range(steps):
                start = (i*batch_size) % self.dataset_size
                end = min(start+batch_size, self.dataset_size)
                sess.run(self.step, feed_dict={self.x: X[start: end], self.y_: Y[start: end]})
                if i % 1000 == 0:
                    total_losses = sess.run(self.cross_entropy, feed_dict={self.x: X, self.y_: Y})
                    logging.info("After {} training steps, crosses entropy on all data is {}".format(i, total_losses))

    def predict(self):
        pass

    def generate_data(self, size=128, rdm_seed=1):
        r = rdm(rdm_seed)
        self.dataset_size = size
        X = r.rand(size, self.x_dimension)
        Y = [[int(sum(xs) < self.x_dimension/2)] for xs in X]
        return X, Y


if __name__ == '__main__':
    mlp = MLP(in_size=2, out_size=1)
    mlp.train()
