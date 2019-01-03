#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/23 8:25 PM
# @Author  : zhangzhen
# @Site    : 
# @File    : torch_cnn.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1)

# hyper parameters
EPOCH = 1
BATCH_SIZE = 128
LR = 0.001
DOWNLOAD_MNIST = False

root = '/Users/zhangzhen/gitRepository/AnalyticsVidhya/data/mnist'
train_data = torchvision.datasets.MNIST(
    root=root,
    train=True,
    transform=torchvision.transforms.ToTensor(),    # transform -> [0, 1]
    download=DOWNLOAD_MNIST
)

# print pict one
# print(train_data.train_data.size())     # (60000, 28, 28)
# print(train_data.train_labels.size())   # (60000,)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_data = torchvision.datasets.MNIST(
    root=root,
    train=False,
)

test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000]
print(test_x.size(), test_y.size())


# define CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # (1, 28, 28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),  # -> (16, 28, 28)
            nn.ReLU(),  # -> (16, 28, 28)
            nn.MaxPool2d(kernel_size=2),    # -> (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # -> (32, 14, 14)
            nn.ReLU(),  # -> (32, 14, 14)
            nn.MaxPool2d(kernel_size=2) # -> (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, *input):
        x = self.conv1(input[0])
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # (batch, 32 * 7 * 7)
        output = self.out(x)
        return output


cnn = CNN()
# print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# training step
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            test_output = cnn(test_x)
            _, pred_y = torch.max(test_output, 1)
            acc = sum(pred_y.numpy() == test_y.numpy()) / test_y.size(0)
            print('Epoch:', epoch, '| train loss: %.4f' % loss.item(), 'test acc: %.4f' % acc)

# print 10 predictions from test data
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
