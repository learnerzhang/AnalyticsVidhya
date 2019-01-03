#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-01-03 10:06
# @Author  : zhangzhen
# @Site    : 
# @File    : torch_loss_function.py
# @Software: PyCharm
import torch
import torch.nn.functional as F

"""
    整理目前实现的loss函数
"""


def binary_cross_entropy():
    """
        二分类交叉熵
        loss(y,y_) = −1/2 [y_ * log y + (1 − y_) * log(1 − y_)]
    """
    loss_fn = torch.nn.BCELoss(reduce=False, size_average=False)
    loss_reduce_fn = torch.nn.BCELoss(reduce=True, size_average=False)
    loss_average_fn = torch.nn.BCELoss(reduce=False, size_average=True)
    loss_reduce_average_fn = torch.nn.BCELoss(reduce=True, size_average=True)

    outputs = F.sigmoid(torch.randn(3, 4))

    # targets must have the same shape with outputs
    targets = torch.FloatTensor(3, 4).random_(to=2)

    # targets = torch.LongTensor(3).random_(to=2) # Shape Error
    print(outputs.size(), "\n", targets)

    print("loss(reduce=False, size_average=False):\n\t", loss_fn(outputs, targets))
    print("loss(reduce=True, size_average=False):\n\t", loss_reduce_fn(outputs, targets))
    print("loss(reduce=False, size_average=True):\n\t", loss_average_fn(outputs, targets))
    print("loss(reduce=True, size_average=True):\n\t", loss_reduce_average_fn(outputs, targets))


def softmax_cross_entropy():
    """
        多分类交叉熵
        L(y,y_)= −1/n ∑[y−_ × log2​(y)]
    """
    loss_fn = torch.nn.CrossEntropyLoss(reduce=False, size_average=False)
    loss_reduce_fn = torch.nn.CrossEntropyLoss(reduce=True, size_average=False)
    loss_average_fn = torch.nn.CrossEntropyLoss(reduce=False, size_average=True)
    loss_reduce_average = torch.nn.CrossEntropyLoss(reduce=True, size_average=True)

    outputs = F.softmax(torch.randn(10, 50))
    targets = torch.LongTensor(10).random_(to=11)
    print(outputs.size(), "\n", targets)
    print("loss(reduce=False, size=False):", loss_fn(outputs, targets))
    print("loss(reduce=True, size=False):", loss_reduce_fn(outputs, targets))
    print("loss(reduce=False, size=True):", loss_average_fn(outputs, targets))
    print("loss(reduce=True, size=True", loss_reduce_average(outputs, targets))


if __name__ == '__main__':
    binary_cross_entropy()
    print(100 * "=")
    softmax_cross_entropy()
    print(100 * "=")
