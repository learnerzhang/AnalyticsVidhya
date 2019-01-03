#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/23 10:37 AM
# @Author  : zhangzhen
# @Site    : 
# @File    : torch_nn_basic.py
# @Software: PyCharm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# generate
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
print(x.shape, y.shape)

x, y = Variable(x), Variable(y)
# print(x, y)

# plt.scatter(x.data.numpy(), y.data.numpy(), c='blue')
# plt.show()


class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, *input):
        tmpX = F.relu(self.hidden(input[0]))
        x = self.predict(tmpX)
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)

# plt
plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1,)
loss_fun = torch.nn.MSELoss()

for t in range(1000):
    prediction = net(x)
    loss = loss_fun(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 5 == 0:
        # plt
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
