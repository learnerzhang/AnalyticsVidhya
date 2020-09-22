#!/usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import random
import time
import generate_sample_v1
import generate_sample_v2
import generate_sample_v3

random.seed(9102)

# add bn layer
norm, epsilon = False, 0.001

# input embedding length
INPUT_EMBEDDING_LENGTH = 260

# input file
FILE_TRAIN = 'embedding_1224_input'  
FILE_VALI = 'embedding_vali'
LEARNING_RATE = 0.001
SUMMARIES_DIR = './Summaries/'
#NUM_EPOCH = 50
NUM_EPOCH = 1

# negative sample
NEG = 4
# query batch size
query_BS = 100
# batch size
L1_N = 128
L2_N = 32

# read data
data_train = generate_sample_v2.generate_sample(FILE_TRAIN)
print 'load data train over'
data_vali = generate_sample_v2.generate_sample(FILE_VALI)
print 'load data vali over'
train_epoch_steps = int(len(data_train['query']) / query_BS) - 1
vali_epoch_steps = int(len(data_vali['query']) / query_BS) - 1


def read_and_decode(filename):
	filename_queue = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example, features={
		'query': tf.FixedLenFeature([], tf.float32),
		'query_len': tf.FixedLenFeature([], tf.int64),
		'doc_pos': tf.FixedLenFeature([], tf.float32),
		'doc_pos_len': tf.FixedLenFeature([], tf.int64),
		'doc_neg': tf.FixedLenFeature([], tf.float32),
		'doc_neg_len': tf.FixedLenFeature([], tf.int64)})

	query = tf.cast(features['query'], tf.float32)
	query_len = tf.cast(features['query_len'], tf.float32)
	doc_pos = tf.cast(features['doc_pos'], tf.float32)
	doc_pos_len = tf.cast(features['doc_pos_len'], tf.float32)
	doc_neg = tf.cast(features['doc_neg'], tf.float32)
	doc_neg_len = tf.cast(features['doc_neg_len'], tf.float32)

	return query, query_len, doc_pos, doc_pos_len, doc_neg, doc_neg_len

def add_layer(inputs, in_size, out_size, activation_function=None):
	#print in_size, out_size
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
	doc_positive_batch = tf.placeholder(tf.float32, shape=[None, None], name='doc_positive_batch')
	doc_negative_batch = tf.placeholder(tf.float32, shape=[None, None], name='doc_negative_batch')
	query_seq_length = tf.placeholder(tf.float32, shape=[None], name='query_seq_length')
	pos_seq_length = tf.placeholder(tf.float32, shape=[None], name='pos_seq_length')
	neg_seq_length = tf.placeholder(tf.float32, shape=[None], name='neg_seq_length')
	on_train = tf.placeholder(tf.bool, name='on_train')
	keep_prob = tf.placeholder(tf.float32, name='drop_out_prob')

with tf.name_scope('FC1'):
	query_l1 = add_layer(query_batch, INPUT_EMBEDDING_LENGTH, L1_N, activation_function=None)
	doc_positive_l1 = add_layer(doc_positive_batch, INPUT_EMBEDDING_LENGTH, L1_N, activation_function=None)
	doc_negative_l1 = add_layer(doc_negative_batch, INPUT_EMBEDDING_LENGTH, L1_N, activation_function=None)

with tf.name_scope('BN1'):
	query_l1 = batch_normalization(query_l1, on_train, L1_N)
	doc_positive_l1 = batch_normalization(doc_positive_l1, on_train, L1_N)
	doc_negative_l1 = batch_normalization(doc_negative_l1, on_train, L1_N)

	query_l1 = tf.nn.relu(query_l1)
	doc_positive_l1 = tf.nn.relu(doc_positive_l1)
	doc_negative_l1 = tf.nn.relu(doc_negative_l1)

with tf.name_scope('Drop_out'):
	query_l1 = tf.nn.dropout(query_l1, keep_prob)
	doc_positive_l1 = tf.nn.dropout(doc_positive_l1, keep_prob)
	doc_negative_l1 = tf.nn.dropout(doc_negative_l1, keep_prob)

with tf.name_scope('FC2'):
	query_l2 = add_layer(query_l1, L1_N, L2_N, activation_function=None)
	doc_positive_l2 = add_layer(doc_positive_l1, L1_N, L2_N, activation_function=None)
	doc_negative_l2 = add_layer(doc_negative_l1, L1_N, L2_N, activation_function=None)

with tf.name_scope('BN2'):
	query_l2 = batch_normalization(query_l2, on_train, L2_N)
	doc_positive_l2 = batch_normalization(doc_positive_l2, on_train, L2_N)
	doc_negative_l2 = batch_normalization(doc_negative_l2, on_train, L2_N)
	
	query_l2 = tf.nn.relu(query_l2)
	doc_positive_l2 = tf.nn.relu(doc_positive_l2)
	doc_negative_l2 = tf.nn.relu(doc_negative_l2)

	query_pred = tf.nn.relu(query_l2, name='query_pred')
	doc_positive_pred = tf.nn.relu(doc_positive_l2, name='doc_pred_pos')
	doc_negative_pred = tf.nn.relu(doc_negative_l2, name='doc_pred_neg')

	# query_pred = tf.contrib.slim.batch_norm(query_l2, activation_fn=tf.nn.relu)

#add by guoxu at 2019-12-24
with tf.name_scope('Merge_Negative_Doc'):
	doc_y = tf.tile(doc_positive_l2, [1, 1])
	for i in range(NEG):
		for j in range(query_BS):
			doc_y = tf.concat([doc_y, tf.slice(doc_negative_l2, [j * NEG + i, 0], [1, -1])], 0) 
#end

with tf.name_scope('Cosine_Similarity'):
	# Cosine similarity
	#cos_sim = get_cosine_score(query_pred, doc_pred)
	#cos_sim_prob = tf.clip_by_value(cos_sim, 1e-8, 1.0)
	query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_l2), 1, True)), [NEG + 1, 1])
	doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))
	prod = tf.reduce_sum(tf.multiply(tf.tile(query_l2, [NEG + 1, 1]), doc_y), 1, True)
	norm_prod = tf.multiply(query_norm, doc_norm)
	cos_sim_raw = tf.truediv(prod, norm_prod)
	cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, query_BS])) * 20

with tf.name_scope('Loss'):
	# Train Loss
	#cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=doc_label_batch, logits=cos_sim)
	#losses = tf.reduce_sum(cross_entropy)
	#tf.summary.scalar('loss', losses)
	prob = tf.nn.softmax(cos_sim)
	hit_prob = tf.slice(prob, [0, 0], [-1, 1])
	loss = -tf.reduce_sum(tf.log(hit_prob))
	tf.summary.scalar('loss', loss)

with tf.name_scope('Training'):
	# Optimizer
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
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

def pull_batch(data_map, batch_id):
	query_in = data_map['query'][batch_id * query_BS:(batch_id + 1) * query_BS]
	query_len = data_map['query_len'][batch_id * query_BS:(batch_id + 1) * query_BS]
	doc_positive_in = data_map['doc_pos'][batch_id * query_BS:(batch_id + 1) * query_BS]
	doc_positive_len = data_map['doc_pos_len'][batch_id * query_BS:(batch_id + 1) * query_BS]
	doc_negative_in = data_map['doc_neg'][batch_id * query_BS * NEG:(batch_id + 1) * query_BS * NEG]
	doc_negative_len = data_map['doc_neg_len'][batch_id * query_BS * NEG:(batch_id + 1) * query_BS * NEG]
	return query_in, doc_positive_in, doc_negative_in, query_len, doc_positive_len, doc_negative_len	

def feed_dict(on_training, data_set, batch_id, drop_prob):
	query_in, doc_positive_in, doc_negative_in, query_seq_len, pos_seq_len, neg_seq_len = pull_batch(data_set, batch_id)
	query_len = len(query_in)
	query_seq_len = [INPUT_EMBEDDING_LENGTH] * query_len
	pos_seq_len = [INPUT_EMBEDDING_LENGTH] * query_len
	neg_seq_len = [INPUT_EMBEDDING_LENGTH] * query_len * NEG
	#print np.shape(query_in), np.shape(doc_positive_in), np.shape(doc_negative_in)
	return {query_batch: query_in, doc_positive_batch: doc_positive_in, doc_negative_batch: doc_negative_in, on_train: on_training, keep_prob: drop_prob, query_seq_length: query_seq_len, neg_seq_length: neg_seq_len, pos_seq_length: pos_seq_len}

saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	train_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/train', sess.graph)

	start = time.time()
	for epoch in range(NUM_EPOCH):
		batch_ids = [i for i in range(train_epoch_steps)]
		random.shuffle(batch_ids)
		for batch_id in batch_ids:
			# print(batch_id)
			sess.run(train_step, feed_dict=feed_dict(True, data_train, batch_id, 0.5))
			pass

		end = time.time()
		# train loss
		epoch_loss = 0
		for i in range(train_epoch_steps):
			loss_v = sess.run(loss, feed_dict=feed_dict(False, data_train, i, 1))
			epoch_loss += loss_v

		epoch_loss /= (train_epoch_steps)
		train_loss = sess.run(train_loss_summary, feed_dict={train_average_loss: epoch_loss})
		train_writer.add_summary(train_loss, epoch + 1)
		print("\nEpoch #%d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" % (epoch, epoch_loss, end - start))

		# test loss
		start = time.time()
		epoch_loss = 0
		for i in range(vali_epoch_steps):
			loss_v = sess.run(loss, feed_dict=feed_dict(False, data_vali, i, 1))
			epoch_loss += loss_v
		epoch_loss /= (vali_epoch_steps)
		test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
		train_writer.add_summary(test_loss, epoch + 1)
		# test_writer.add_summary(test_loss, step + 1)
		print("Epoch #%d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs" % (epoch, epoch_loss, start - end))

	save_path = saver.save(sess, "model_2/model_1.ckpt")
	print("Model saved in file: ", save_path)
