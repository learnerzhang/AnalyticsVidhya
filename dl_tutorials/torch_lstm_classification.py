#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/24 10:05 AM
# @Author  : zhangzhen
# @Site    : 
# @File    : torch_lstm_classification.py
# @Software: PyCharm
import torch
import torchvision
from torch import nn
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision.datasets as ds
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

torch.manual_seed(1)  # reproducible

# hyper Parameters
EPOCH = 1
BATCH_SIZE = 128
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False
ROOT = '/Users/zhangzhen/gitRepository/AnalyticsVidhya/data/mnist'
train_data = torchvision.datasets.MNIST(
    root=ROOT,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

# plt.imshow(train_data.train_data[0], cmap='gray')
# plt.show()
test_data = torchvision.datasets.MNIST(
    root=ROOT,
    train=False,
)

# batch setting
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)
# simplify data set
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[: 2000]/255.
test_y = test_data.test_labels.numpy().squeeze()[: 2000]

print(test_x.shape, test_y.shape)


class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=2,
            batch_first=True,   # False -> (time_step, batch, input)  True -> (batch, time_step, input)
        )
        self.out = nn.Linear(64, 10)

    def forward(self, *input):
        r_out, h_state = self.rnn(input[0], None)  # x-> (batch, time_step, input_size)
        out = self.out(r_out[:, -1, :]) # (batch, time_step, input)
        return out


if __name__ == '__main__':

    rnn = RNN()
    # print(rnn)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x.view(-1, 28, 28))
            b_y = Variable(y)
            output = rnn(b_x)
            loss = loss_func(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_out = rnn(test_x)
                pred_y = torch.max(test_out, 1)[1].data.numpy().squeeze()
                acc = sum(pred_y==test_y)/test_y.size
                print('Epoch:', epoch, '| train loss: %.4f' % loss.item(), 'test acc: %.4f' % acc)

    # print 10 predictions from test data
    test_output = rnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10], 'real number')
