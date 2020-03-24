# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/3/13  21:51
"""
# LSTM
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)  # input dim is 3, and output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
# print(hidden)
for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), hidden)
inputs = torch.cat(inputs).view(len(inputs),1,-1)
hidden = (torch.randn(1,1,3),torch.randn(1,1,3))
out,hidden = lstm(inputs,hidden)
print(out)
print(hidden)