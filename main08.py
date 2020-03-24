# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/3/13  19:16
"""
# Pytorch实现基本循环神经网络RNN

import torch
import torch.nn as nn
rnn = nn.RNN(10,20,2)
input = torch.randn(5,3,10)
h0 = torch.randn(2,3,20)
output,hn = rnn(input,h0)
print(output)
print(hn)


