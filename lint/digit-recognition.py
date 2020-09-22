#!/usr/bin/python
import pandas as pd
from sklearn import svm


def get_data(dim=1):
    df = pd.read_csv("ml_problems_1_train.csv")
    labels = []
    features = []
    for index, row in df.iterrows():
        tmp = list(row)
        if dim == 1:
            labels.append(tmp[0])
            features.append(tmp[1:])
        elif dim == 2:
            t_label = tmp[0]
            t_feature = []
            for i in range(0,28):
                feat = []
                for j in range(0,28):
                     feat.append(tmp[i * 28 + j + 1])
                t_feature.append(feat)
            features.append(t_feature)
            labels.append(t_label)

    return features, labels


def run():
    trainX, trainY = get_data(dim=1)
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(trainX, trainY)


run()