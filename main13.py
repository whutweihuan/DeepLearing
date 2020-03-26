# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/3/26  21:03
"""
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
BATCH_SIZE = 258

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


model = torchvision.models.vgg16().to(device)