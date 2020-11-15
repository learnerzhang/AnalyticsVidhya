#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/20 10:45 AM
# @Author  : zhangzhen
# @Site    : 
# @File    : basic_nn_classification.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
print(tf)

FEATURE_NUM = 32  # 特征数量
HIDDEN_NUM = 64  # 隐藏层
LEARNING_RATE = 0.02  # 学习速率
THRESHOLD = 0.5  # 判断门限

# 清空 Graph
tf.compat.v1.reset_default_graph()

with tf.name_scope("input"):
	# input
	x = tf.compat.v1.placeholder(tf.float32, shape=[None, FEATURE_NUM])
	y = tf.compat.v1.placeholder(tf.int32, shape=[None])
	y_expand = tf.expand_dims(y, axis=1)


with tf.name_scope("fm"):
	# weight
	bais = tf.Variable([0.0])

	# first factor
	weight = tf.Variable(tf.random.uniform([FEATURE_NUM, 1], 0.0, 1.0))

	# second factor
	weight_mix = tf.Variable(tf.random.uniform([FEATURE_NUM, HIDDEN_NUM], 0.0, 1.0))

	BATCH_SIZE = tf.shape(x)[0]

	# BATCH_SIZE * FEATURE_N * 1
	tmp_x = tf.reshape(x, [BATCH_SIZE, FEATURE_NUM, 1])

	# BATCH_SIZE * FEATURE_N * HIDDEN_N
	x_mix_input = tf.tile(tmp_x, [1, 1, HIDDEN_NUM])
	print("input-x:", x, tmp_x, x_mix_input)

	# 1 * FEATURE_N * HIDDEN_N
	w_tmp = tf.reshape(weight_mix, [1, FEATURE_NUM, HIDDEN_NUM])
	# BATCH_SIZE * FEATURE_N * HIDDEN_N
	w_mix_input = tf.tile(w_tmp, [BATCH_SIZE, 1, 1])
	print("weight:", weight_mix, w_tmp, w_mix_input)


	# BATCH_SIZE * FEATURE_N * HIDDEN_N
	embeddings = tf.multiply(x_mix_input, w_mix_input)

	# BATCH_SIZE * FEATURE_N * HIDDEN_N
	embeddings_square = tf.square(embeddings)

	# BATCH_SIZE * HIDDEN_N
	embeddings_square_sum = tf.reduce_sum(embeddings_square, 1)
	print("mix-result:", embeddings_square, embeddings_square_sum)

	# BATCH_SIZE * HIDDENT_N
	embeddings_sum = tf.reduce_sum(embeddings, 1)

	# BATCH_SIZE * HIDDENT_N
	embeddings_sum_square = tf.square(embeddings_sum)
	print("rare-result:", embeddings_sum, embeddings_sum_square)

	z = bais + tf.matmul(x, weight) + 1.0 / 2.0 * tf.reduce_sum(tf.subtract(embeddings_sum_square, embeddings_square_sum), 1, keepdims=True)
	hypothesis = tf.sigmoid(z)

	print("hypothesis:", hypothesis)


with tf.name_scope("loss"):
	loss = tf.compat.v1.losses.log_loss(y_expand, hypothesis)
	print("loss:", loss)


with tf.name_scope("train"):
	optimizer = tf.compat.v1.train.GradientDescentOptimizer(LEARNING_RATE)
	training_op = optimizer.minimize(loss)


with tf.name_scope("eval"):
	predictions = tf.compat.v1.to_int32(hypothesis - THRESHOLD)
	corrections = tf.equal(predictions, y_expand)
	accuracy = tf.reduce_mean(tf.cast(corrections, tf.float32))


# 初始化所有变量
init = tf.compat.v1.global_variables_initializer()


EPOCH = 10  # 迭代次数
with tf.compat.v1.Session() as sess:
	sess.run(init)
	for i in range(EPOCH):
		_training_op, _loss = sess.run([training_op, loss], feed_dict={x: np.random.rand(10, FEATURE_NUM), y: np.random.randint(2, size=10)})
		_accuracy = sess.run([accuracy], feed_dict={x: np.random.rand(5, FEATURE_NUM), y: np.random.randint(2, size=5)})
		print("epoch:", i, _loss, _accuracy)

if __name__ == '__main__':
	pass
