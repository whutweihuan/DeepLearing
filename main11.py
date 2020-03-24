# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/3/24  18:53
"""
# 简单的神经网络，预测异或,读取模型
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
       super(Model,self).__init__()
       self.fc1 = nn.Linear(2,20,bias = True)
       self.fc2 = nn.Linear(20,1,bias = True)

    def forward(self,x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
model = torch.load('xor.pkl')
# print(model)
x = torch.FloatTensor([[1,0],[1,1],[0,0],[0,1]])
y_pred = model(x.cuda())
print(y_pred)

# for i in range(len(x)):
#     print(x[i,:],y_pred[i])

