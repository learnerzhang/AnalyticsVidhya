#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/23 10:01 AM
# @Author  : zhangzhen
# @Site    : 
# @File    : torch_tutorial.py
# @Software: PyCharm
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt


def act_functions():
    """
    draw the activation function
    :return:
    """
    # generate data
    x = torch.linspace(-5, 5, 200)
    x = Variable(x)

    x_np = x.data.numpy()

    y_relu = F.relu(x).data.numpy()
    y_sigmoid = F.sigmoid(x).data.numpy()
    y_tanh = F.tanh(x).data.numpy()
    y_softplus = F.softplus(x).data.numpy()

    plt.figure(1, figsize=(10, 8))

    # sub fig -- relu
    plt.subplot(221)
    plt.plot(x_np, y_relu, c='red', label='relu')
    plt.ylim(-1, 5)
    plt.legend(loc='best')

    # sub fig -- sigmoid
    plt.subplot(222)
    plt.plot(x_np, y_sigmoid, c='blue', label='sigmoid')
    plt.ylim(-0.2, 1.2)
    plt.legend(loc='best')

    # sub fig -- tanh
    plt.subplot(223)
    plt.plot(x_np, y_tanh, c='green', label='tanh')
    plt.ylim(-1.2, 1.2)
    plt.legend(loc='best')

    plt.subplot(224)
    plt.plot(x_np, y_softplus, c='cyan', label='softplus')
    plt.ylim(-0.2, 6)
    plt.legend(loc='best')

    plt.show()


def loss_functions():
    # head
    plt.figure(figsize=(20, 16))
    plt.title(u"损失函数")

    # sub fig -- sigmoid
    plt.subplot(331)
    x_sig = np.linspace(start=-15, stop=15, dtype=np.float)
    loss_sigmoid = 1 / (1 + np.exp(-x_sig))

    plt.plot(x_sig, loss_sigmoid, c='red', label='sigmoid')
    plt.ylim(-0.2, 1.2)
    plt.legend(loc='best')
    plt.grid()

    # sub fig -- logistics
    plt.subplot(332)
    x_logi = np.linspace(start=-5, stop=10, dtype=np.float)
    loss_logistics = np.log((1 + np.exp(-x_logi))) / np.log(2)

    plt.plot(x_logi, loss_logistics, c='blue', label='logistics')
    plt.ylim(-0.5, 8)
    plt.legend(loc='best')
    plt.grid()

    # sub fig -- boost
    plt.subplot(333)
    x_boost = np.linspace(start=-3, stop=10, dtype=np.float)
    loss_boost = np.exp(-x_boost)

    plt.plot(x_boost, loss_boost, c='cyan', label='boost')
    plt.ylim(-0.2, 15)
    plt.legend(loc='best')
    plt.grid()

    # sub fig -- 0/1
    plt.subplot(334)
    x_0_1 = np.linspace(start=-10, stop=10, num=1001, dtype=np.float)
    loss_0_1 = x_0_1 < 0

    plt.plot(x_0_1, loss_0_1, c='olive', label='0/1')
    plt.ylim(-0.2, 1.2)
    plt.legend(loc='best')
    plt.grid()

    # sub fig -- hinge
    plt.subplot(335)
    x_hinge = np.linspace(-5, 10, num=1001, dtype=np.float)
    loss_hinge = 1.0 - x_hinge
    loss_hinge[loss_hinge < 0] = 0

    plt.plot(x_hinge, loss_hinge, c='navy', label='hinge')
    plt.ylim(-0.2, 5)
    plt.legend(loc='best')
    plt.grid()

    # sub fig --  mse and mae
    plt.subplot(336)
    x_mse_mae = np.linspace(-2.5, 2.5, num=1001, dtype=np.float)
    loss_mse = np.square(x_mse_mae)
    loss_mae = np.abs(x_mse_mae)

    plt.plot(x_mse_mae, loss_mse, c='yellowgreen', label='mse')
    plt.plot(x_mse_mae, loss_mae, c='tan', label='mae')
    plt.ylim(-0.2, 4)
    plt.legend(loc='upper right')
    plt.grid()

    # sub fig -- log cosh
    plt.subplot(337)
    x_log_cosh = np.linspace(-4, 4, num=1001, dtype=np.float)
    loss_log_cosh = np.log2(np.cosh(x_log_cosh))

    plt.plot(x_log_cosh, np.cos(x_log_cosh), c='olivedrab', label='cos')
    plt.plot(x_log_cosh, np.cosh(x_log_cosh), c='maroon', label='cosh')
    plt.plot(x_log_cosh, np.log2(np.cosh(x_log_cosh)), c='palegreen', label='logcosh')
    plt.ylim(-1.5, 10)
    plt.legend(loc='upper right')
    plt.grid()

    # sub fig -- huber
    plt.subplot(338)
    x_huber = np.linspace(-100, 100, num=1001, dtype=np.float)
    plt.plot(x_huber, np.square(x_huber) / 2, c='violet', label='squared loss', lw=2)
    for d in (10, 5, 3, 2, 1):
        plt.plot(x_huber, (abs(x_huber) <= d) * x_huber ** 2 / 2 + (abs(x_huber) > d) * d * (abs(x_huber) - d / 2), label=r'huber loss: $\delta$={}'.format(d), lw=2)

    plt.ylim(-1, 1000)
    plt.legend(loc='upper right')
    plt.grid()


    # sub fig -- all loss function
    plt.subplot(339)
    x = np.linspace(-2, 3, 1001, dtype=float)
    plt.plot(x, np.log((1 + np.exp(-x))) / np.log(2), 'r--', label='Logistics Loss', lw=2)
    plt.plot(x, np.exp(-x), 'k-', label='Adaboost Loss', lw=1)
    plt.plot(x, x < 0, 'y-', label='0/1 Loss', lw=1)
    tmp_hinge = 1.0 - x
    tmp_hinge[tmp_hinge < 0] = 0
    plt.plot(x, tmp_hinge, 'b-', label='Hinge Loss', lw=1)
    plt.legend(loc='best')
    plt.grid()

    # save
    plt.savefig("loss_function.png")
    # show
    plt.show()


if __name__ == '__main__':
    # act_functions()
    loss_functions()
