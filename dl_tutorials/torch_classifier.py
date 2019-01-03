#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-12-24 17:28
# @Author  : zhangzhen
# @Site    : 
# @File    : torch_classifier.py
# @Software: PyCharm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random

ROOT = '/Users/zhangzhen/gitRepository/AnalyticsVidhya/data/cifar'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
print(transforms)

trainset = torchvision.datasets.CIFAR10(
    root=ROOT,
    train=True,
    download=False,
    transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root=ROOT,
    train=False,
    download=False,
    transform=transform
)
print(trainset.train_data.shape, len(trainset.train_labels))

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def plot_figs(num_figs=10):
    fig, axes = plt.subplots(1, num_figs, figsize=(1 * num_figs, 1))
    num_train = len(trainset.train_labels)
    random_nums = random.sample(range(num_train), num_figs)
    for i in range(num_figs):
        axes[i].imshow(trainset.train_data[random_nums[i]])
        axes[i].set_title(classes[trainset.train_labels[random_nums[i]]])
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.show()


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# plot_figs(num_figs=4)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=5,
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=5,
        )

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, *input):
        x = F.max_pool2d(F.relu(self.conv1(input[0])), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = ConvNet()
print(net)

criteria = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criteria(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 2000 == 1999:
            print("[{}, {}], Loss: {}".format(epoch + 1, i + 1, running_loss / 2000.))
            running_loss = 0.0

print("Finished Training")


dataiter = iter(testloader)
images, labels = dataiter.next()

outputs = net(images)
_, predicted = torch.max(outputs, 1)

imshow(torchvision.utils.make_grid(images))
for i in range(4):
    print("Actual {}\tPredicted: {}".format(classes[labels[i]], classes[predicted[i]]))

