# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/3/26  15:03
"""
# 简单图片图片分类，针对的是CIFAR

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# input()
# print(device)
BATCH_SIZE = 512

trans = torchvision.transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4733649,), (0.25156906,)), ])

train_loader = DataLoader(datasets.CIFAR10('CIFAR10', train = True, download = False,
                                           transform = trans,
                                           target_transform = None),
                          batch_size = BATCH_SIZE,
                          shuffle = True
                          )

test_loader = DataLoader(datasets.CIFAR10('CIFAR10', train = False, download = False,
                                          transform = trans,
                                          target_transform = None),
                         batch_size = BATCH_SIZE,
                         shuffle = True
                         )

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# train_data = data

# data_img =  [d[0].data.cpu().numpy() for d in data]
# print(data[0][0].shape)
# print(np.mean(data_img),np.std(data_img)) # 0.4733649 0.2515690
# print(len(data_img))


class Net(nn.Module):
    def __init__ (self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward (self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # x = F.softmax(x)

        return F.log_softmax(x, dim = 1)


model = Net().to(device)
lr = 0.01
momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = momentum)
# loss_fn = F.cross_entropy()

N_EPOCHS = 200


# print(model)
# input()

# print(train_loader[0][0].shape)
def train ():
    model.train()
    print('---------train-------------')
    for i in range(N_EPOCHS):
        # train_data = train_loader[]
        # train_data 32 x 3 x 32 x 32
        # pred 100 x 10
        # target 100
        best_loss = 1000
        for train_data, target in train_loader:
            # print(train_data,target)
            train_data, target = train_data.to(device), target.to(device)
            # print(train_data.shape)
            pred = model(train_data)
            # print(target.shape)
            optimizer.zero_grad()
            loss = F.nll_loss(pred, target)
            if best_loss > loss.item():
                best_loss = loss.item()
                torch.save(model.state_dict(), 'cifar.pkl')

            loss.backward()
            optimizer.step()
            # input("----------")
        print("Process: {:.3f}% loss: {:.6f}".format((i + 1) / N_EPOCHS * 100, loss.item()))


def test ():
    model.load_state_dict(torch.load('cifar.pkl'))
    model.eval()
    print('---------test-------------')
    for i in range(1):
        # train_data = train_loader[]
        # train_data 100 x 3 x 32 x 32
        # pred 100 x 10
        # target 100
        correct = 0
        for train_data, target in test_loader:
            # print(train_data,target)
            with torch.no_grad():
                train_data, target = train_data.to(device), target.to(device)
                # print(train_data.shape)
                pred = model(train_data)
                pred_label = pred.argmax(dim = 1, keepdim = True)
                # print(pred_label.shape)
                correct += pred_label.eq(target.view_as(pred_label)).sum().item()
            # print(target.shape)
            # optimizer.zero_grad()
            # loss = F.nll_loss(pred,target)
            # loss.backward()
            # optimizer.step()
            # input("----------")
        print("Process: {:.3f}% Accuracy: {:.3f}%".format(i / N_EPOCHS * 100, correct / len(test_loader.dataset) * 100))


def test_mypic ():
    print('--------fact image---------')
    model.load_state_dict(torch.load('cifar.pkl'))
    model.eval()
    img = Image.open('cat2.jpg')
    img = trans(img).to(device)
    img = img.unsqueeze(0)
    pred = model(img)
    # print(pred)
    lable_idx = pred.argmax()
    print(classes[lable_idx])



if __name__ == '__main__':
    # train()
    test()
    test_mypic()
