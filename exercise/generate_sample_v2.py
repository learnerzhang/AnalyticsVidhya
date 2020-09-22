#!/usr/bin/env python
# encoding : utf-8

import sys
import random
reload(sys)

sys.setdefaultencoding('utf-8')
import tensorflow as tf
NEG = 4

##click is positive, no_click is negtive, not use label, pos:neg = 1:4
def generate_sample(file_name):
    list_sample_query = []
    list_sample_positive = []
    list_sample_negative = []
    dict_uid_info = {}
    cur_uid = ''
    list_positive_embedding = []
    list_negative_embedding = []
    list_query_embedding = []

    index = 0
    for line in open(file_name):
        index += 1
        if index % 10000 == 0:
            print index
        temp_list = []
        temp_list = line.strip().split('\t')
        if len(temp_list) != 5:
            continue
        uid, mid, label, uid_feature, mid_feature = temp_list[0:5]
        uid_embedding = uid_feature.strip().split(',')
        uid_embedding = map(float, uid_embedding)
        mid_embedding = mid_feature.strip().split(',')
        mid_embedding = map(float, mid_embedding)
        if label == '1':
            label = 1
        else:
            label = 0

        if cur_uid == '':
            cur_uid = uid
            if label == 1:
                list_query_embedding.append(uid_embedding)
                list_positive_embedding.append(mid_embedding)
            else:
                list_negative_embedding.append(mid_embedding)
        else:
            if cur_uid == uid:
                if label == 1:
                    list_query_embedding.append(uid_embedding)
                    list_positive_embedding.append(mid_embedding)
                else:
                    list_negative_embedding.append(mid_embedding)
            else:
                # print len(list_query_embedding), len(list_positive_embedding), len(list_negative_embedding)
                if len(list_negative_embedding) > 0 and len(list_positive_embedding) > 0:
                    if len(list_negative_embedding) >= NEG * len(list_positive_embedding):
                        random.shuffle(list_negative_embedding)
                        list_negative_embedding = list_negative_embedding[:NEG * len(list_positive_embedding)]
                    else:
                        while len(list_negative_embedding) < NEG * len(list_positive_embedding):
                            rand_index = random.randint(0, len(list_negative_embedding) - 1)
                            list_negative_embedding.append(list_negative_embedding[rand_index])

                    list_sample_query.extend(list_query_embedding)
                    list_sample_positive.extend(list_positive_embedding)
                    list_sample_negative.extend(list_negative_embedding)

                cur_uid = uid
                list_query_embedding = []
                list_positive_embedding = []
                list_negative_embedding = []
                if label == 1:
                    list_query_embedding.append(uid_embedding)
                    list_positive_embedding.append(mid_embedding)
                else:
                    list_negative_embedding.append(mid_embedding)

    # print len(list_query_embedding), len(list_positive_embedding), len(list_negative_embedding)
    if len(list_negative_embedding) > 0 and len(list_positive_embedding) > 0:
        if len(list_negative_embedding) >= NEG * len(list_positive_embedding):
            random.shuffle(list_negative_embedding)
            list_negative_embedding = list_negative_embedding[:NEG * len(list_positive_embedding)]
        else:
            while len(list_negative_embedding) < NEG * len(list_positive_embedding):
                rand_index = random.randint(0, len(list_negative_embedding) - 1)
                list_negative_embedding.append(list_negative_embedding[rand_index])

        list_sample_query.extend(list_query_embedding)
        list_sample_positive.extend(list_positive_embedding)
        list_sample_negative.extend(list_negative_embedding)

    vec_len = len(list_sample_query[0])

    list_query_len = []
    for i in range(len(list_sample_query)):
        list_query_len.append(vec_len)

    list_pos_len = []
    for i in range(len(list_sample_positive)):
        list_pos_len.append(vec_len)

    list_neg_len = []
    for i in range(len(list_sample_negative)):
        list_neg_len.append(vec_len)

    dict_return = {}
    dict_return['query'] = list_sample_query
    dict_return['query_len'] = list_query_len
    dict_return['doc_pos'] = list_sample_positive
    dict_return['doc_pos_len'] = list_pos_len
    dict_return['doc_neg'] = list_sample_negative
    dict_return['doc_neg_len'] = list_neg_len

    return dict_return

def text_to_tfrecord(data_map, tfrecord_path):
    # 第一步：生成TFRecord Writer
    writer = tf.python_io.TFRecordWriter(tfrecord_path)
    nums = len(data_map['query'])
    for i in range(nums):
        query_in = data_map['query'][i]
        query_len = data_map['query_len'][i]
        doc_positive_in = data_map['doc_pos'][i]
        doc_positive_len = data_map['doc_pos_len'][i]
        doc_negative_in = data_map['doc_neg'][i * NEG:(i + 1) * NEG]
        doc_negative_len = data_map['doc_neg_len'][i * NEG:(i + 1) * NEG]

        # 第三步： 建立feature字典，tf.train.Feature()对单一数据编码成feature
        feature = {
            'query': tf.train.Feature(float_list=tf.train.FloatList(value=query_in)),
            'query_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[query_len])),
            'doc_pos': tf.train.Feature(float_list=tf.train.FloatList(value=doc_positive_in)),
            'doc_pos_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[doc_positive_len])),
            'doc_neg': tf.train.Feature(float_list=tf.train.FloatList(value=doc_negative_in)),
            'doc_neg_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[doc_negative_len])),
        }
        # 第四步：可以理解为将内层多个feature的字典数据再编码，集成为features
        features = tf.train.Features(feature=feature)
        # 第五步：将features数据封装成特定的协议格式
        example = tf.train.Example(features=features)
        # 第六步：将example数据序列化为字符串
        Serialized = example.SerializeToString()
        # 第七步：将序列化的字符串数据写入协议缓冲区
        writer.write(Serialized)
    writer.close()

if __name__ == '__main__':
    dict_info = generate_sample(sys.argv[1])
    print len(dict_info['query']), len(dict_info['doc_pos']), len(dict_info['doc_neg'])
    #batch_()
