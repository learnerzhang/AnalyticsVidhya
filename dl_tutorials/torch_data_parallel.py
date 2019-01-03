#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-01-02 20:32
# @Author  : zhangzhen
# @Site    : 
# @File    : torch_data_parallel.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100


class RandomDataSet(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.len


rand_loader = DataLoader(dataset=RandomDataSet(input_size, data_size), batch_size=batch_size, shuffle=True)


class Model(nn.Module):

    def __init__(self, input_size, out_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, out_size)

    def forward(self, *input):
        output = self.fc(input[0])
        print("\tIn Model: input size", input[0].size(), "output size", output.size())

        return output


model = Model(input_size, output_size)

for data in rand_loader:
    output = model(torch.Tensor(data))
    print("Outside: input size", data.size(), "output_size", output.size(), "\n\n")