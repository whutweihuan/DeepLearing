# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/3/24  16:55
"""
# 简单的神经网络，预测异或
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

np.random.seed(1)
torch.random.manual_seed(1)
N_SAMPLE = 100
INPUT = 2
OUTPUT = 1
HIDDEN = 20
BATCH_SIZE = 20
N_STEP = 100000
CUDA_OK = False

x = torch.randint(0,2,(N_SAMPLE,INPUT)).cuda() if CUDA_OK else torch.randint(0,2,(N_SAMPLE,INPUT))
y = torch.zeros(N_SAMPLE,dtype=torch.int32).cuda() if CUDA_OK else torch.zeros(N_SAMPLE,dtype=torch.int32)
y = x[:,0]+x[:,1]
y[y!=1]=0
y = y.view(100,1)
# print(y)
# print(x.data.item())
# print(torch.cuda.is_available())



if torch.cuda.is_available():
    CUDA_OK = True


class Model(nn.Module):
    def __init__(self):
       super(Model,self).__init__()
       self.fc1 = nn.Linear(INPUT,HIDDEN,bias = True)
       self.fc2 = nn.Linear(HIDDEN,OUTPUT,bias = True)

    def forward(self,x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
model = Model().cuda() if CUDA_OK else Model()
optimizer = optim.Adam(model.parameters(),lr=0.05)
loss_fn = nn.MSELoss().cuda() if CUDA_OK else nn.MSELoss()

def train():
    # roop N_STEP times
    for  i in range(N_STEP):
        # do a predict
        y_pred = model(x.cuda())
        # print(y.shape)
        # print(y_pred.shape)
        # use loss funcion
        loss = loss_fn(y_pred.cuda(),y.cuda())
        print(i,loss.item())

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # update model parameters
        optimizer.step()
train()
torch.save(model,'xor.pkl')
# model = torch.load('xor.pkl')

x = torch.FloatTensor([[1,0],[1,1],[0,0],[0,1]])
with torch.no_grad():
    y_pred = model(x.cuda())
    print(y_pred)
