#!/usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import random
import time
import generate_sample_v1

random.seed(9102)

# add bn layer
norm, epsilon = False, 0.001
INPUT_EMBEDDING_LENGTH = 260

FILE_PREDICT='ceshi'
LEARNING_RATE = 0.001
SUMMARIES_DIR = './Summaries/'
NUM_EPOCH = 50

# negative sample
# query batch size
query_BS = 100
# batch size
L1_N = 128
L2_N = 32

# read data
data_predict = generate_sample_v1.generate_sample(FILE_PREDICT)
predict_epoch_steps = int(len(data_predict) / query_BS) - 1

def add_layer(inputs, in_size, out_size, activation_function=None):
	wlimit = np.sqrt(6.0 / (in_size + out_size))
	Weights = tf.Variable(tf.random_uniform([in_size, out_size], -wlimit, wlimit))
	biases = tf.Variable(tf.random_uniform([out_size], -wlimit, wlimit))
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs

def mean_var_with_update(ema, fc_mean, fc_var):
	ema_apply_op = ema.apply([fc_mean, fc_var])
	with tf.control_dependencies([ema_apply_op]):
		return tf.identity(fc_mean), tf.identity(fc_var)

def batch_normalization(x, phase_train, out_size):
	"""
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        out_size:    integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """

	with tf.variable_scope('bn'):
		beta = tf.Variable(tf.constant(0.0, shape=[out_size]), name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[out_size]), name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed


def variable_summaries(var, name):
	"""Attach a lot of summaries to a Tensor."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean/' + name, mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
		tf.summary.scalar('sttdev/' + name, stddev)
		tf.summary.scalar('max/' + name, tf.reduce_max(var))
		tf.summary.scalar('min/' + name, tf.reduce_min(var))
		tf.summary.histogram(name, var)

def contrastive_loss(y, d, batch_size):
	tmp = y * tf.square(d)
	# tmp= tf.mul(y,tf.square(d))
	tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
	reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
	return tf.reduce_sum(tmp + tmp2) / batch_size / 2 + reg

def get_cosine_score(query_arr, doc_arr):
	# query_norm = sqrt(sum(each x^2))
	pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(query_arr), 1))
	pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(doc_arr), 1))
	pooled_mul_12 = tf.reduce_sum(tf.multiply(query_arr, doc_arr), 1)
	cos_scores = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="cos_scores")
	return cos_scores

with tf.name_scope('input'):
	query_batch = tf.placeholder(tf.float32, shape=[None, None], name='query_batch')
	doc_batch = tf.placeholder(tf.float32, shape=[None, None], name='doc_batch')
	doc_label_batch = tf.placeholder(tf.float32, shape=[None], name='doc_label_batch')
	on_train = tf.placeholder(tf.bool)
	keep_prob = tf.placeholder(tf.float32, name='drop_out_prob')

with tf.name_scope('FC1'):
	query_l1 = add_layer(query_batch, INPUT_EMBEDDING_LENGTH, L1_N, activation_function=None)
	doc_l1 = add_layer(doc_batch, INPUT_EMBEDDING_LENGTH, L1_N, activation_function=None)

with tf.name_scope('BN1'):
	query_l1 = batch_normalization(query_l1, on_train, L1_N)
	doc_l1 = batch_normalization(doc_l1, on_train, L1_N)
	query_l1 = tf.nn.relu(query_l1)
	doc_l1 = tf.nn.relu(doc_l1)

with tf.name_scope('Drop_out'):
	query_l1 = tf.nn.dropout(query_l1, keep_prob)
	doc_l1 = tf.nn.dropout(doc_l1, keep_prob)

with tf.name_scope('FC2'):
	query_l2 = add_layer(query_l1, L1_N, L2_N, activation_function=None)
	doc_l2 = add_layer(doc_l1, L1_N, L2_N, activation_function=None)

with tf.name_scope('BN2'):
	query_l2 = batch_normalization(query_l2, on_train, L2_N)
	doc_l2 = batch_normalization(doc_l2, on_train, L2_N)
	query_l2 = tf.nn.relu(query_l2)
	doc_l2 = tf.nn.relu(doc_l2)

	query_pred = tf.nn.relu(query_l2)
	doc_pred = tf.nn.relu(doc_l2)

	# query_pred = tf.contrib.slim.batch_norm(query_l2, activation_fn=tf.nn.relu)

with tf.name_scope('Cosine_Similarity'):
	# Cosine similarity
	cos_sim = get_cosine_score(query_pred, doc_pred)
	cos_sim_prob = tf.clip_by_value(cos_sim, 1e-8, 1.0)

with tf.name_scope('Loss'):
	# Train Loss
	cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=doc_label_batch, logits=cos_sim)
	losses = tf.reduce_sum(cross_entropy)
	tf.summary.scalar('loss', losses)
	pass

with tf.name_scope('Training'):
	# Optimizer
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(losses)
	pass

#with tf.name_scope('Accuracy'):
#	correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
#	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#	tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.name_scope('Test'):
	average_loss = tf.placeholder(tf.float32)
	loss_summary = tf.summary.scalar('average_loss', average_loss)

with tf.name_scope('Train'):
	train_average_loss = tf.placeholder(tf.float32)
	train_loss_summary = tf.summary.scalar('train_average_loss', train_average_loss)

def pull_all(query_in, doc_positive_in, doc_negative_in):
	query_in = query_in.tocoo()
	doc_positive_in = doc_positive_in.tocoo()
	doc_negative_in = doc_negative_in.tocoo()

	query_in = tf.SparseTensorValue(np.transpose([np.array(query_in.row, dtype=np.int64), np.array(query_in.col, dtype=np.int64)]), np.array(query_in.data, dtype=np.float), np.array(query_in.shape, dtype=np.int64))

	doc_positive_in = tf.SparseTensorValue(np.transpose([np.array(doc_positive_in.row, dtype=np.int64), np.array(doc_positive_in.col, dtype=np.int64)]), np.array(doc_positive_in.data, dtype=np.float), np.array(doc_positive_in.shape, dtype=np.int64))

	doc_negative_in = tf.SparseTensorValue(np.transpose([np.array(doc_negative_in.row, dtype=np.int64), np.array(doc_negative_in.col, dtype=np.int64)]), np.array(doc_negative_in.data, dtype=np.float), np.array(doc_negative_in.shape, dtype=np.int64))

	return query_in, doc_positive_in, doc_negative_in

def pull_batch(data_map, batch_id):
	query, title, label, dsize = range(4)
	cur_data = data_map[batch_id * query_BS:(batch_id + 1) * query_BS]
	query_in = [x[0] for x in cur_data]
	doc_in = [x[1] for x in cur_data]
	label = [x[2] for x in cur_data]

	# query_in, doc_positive_in, doc_negative_in = pull_all(query_in, doc_positive_in, doc_negative_in)
	return query_in, doc_in, label

def feed_dict(on_training, data_set, batch_id, drop_prob):
	query_in, doc_in, label = pull_batch(data_set, batch_id)
	query_in, doc_in, label = np.array(query_in), np.array(doc_in), np.array(label)
	return {query_batch: query_in, doc_batch: doc_in, doc_label_batch: label, on_train: on_training, keep_prob: drop_prob}

with tf.Session() as sess:
	saver = tf.train.import_meta_graph('model/model_1.ckpt.meta')
	#saver.restore(sess, tf.train.latest_checkpoint('model/'))
	#graph = tf.get_default_graph()
	
	#query_value = sess.run(query_pred, feed_dict=feed_dict(False, data_predict, 0, 1))	
	#doc_value = sess.run(doc_pred, feed_dict=feed_dict(False, data_predict, 0, 1))	
	#print query_value
