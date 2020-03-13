# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/1/10  22:20
"""
# 这是一份练习文档
from __future__ import print_function
import torch

# 全是 0
x = torch.empty(5, 3)
print(x)

# 0 到 1
x = torch.rand(5, 3)
print(x)

# x = torch.zeros(5,3,dtype = torch.long)
# print(x)

x = torch.tensor([5.5, 3])
print(x)

# x1 = x1.new_ones(5,3,dtype = torch.double)
# print(x1)

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)

import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
print(b)


# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device('cuda')  # a CUDA device object
    y = torch.ones_like(x, device = device)  # directly create a tensor on GPU
    x = x.to(device) # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double)) # ``.to`` can also change dtype together!

