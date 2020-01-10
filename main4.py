# -*- coding: utf-8 -*-
"""
 author: weihuan
 date: 2020/1/10  22:43
"""
# AUTOGRAD: AUTOMATIC DIFFERENTIATION
# 自动求导?自动梯度

import torch

x = torch.ones(2, 2, requires_grad = True)
# print(x)

y = x + 2
# print(y)
# print(y.grad_fn)

z = y * y * 3
out = z.mean()
# print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
# print(a.requires_grad)
a.requires_grad = True
# print(a.requires_grad)
b = (a * a).sum()
# print(b.grad_fn)


# out.backward()
# print(x.grad)

# vector - Jacobian
# product:
x = torch.randn(3, requires_grad = True)
y = x * 2
while y.data.norm() < 2:
    y = y * 2
# print(y)


# v = torch.tensor([0.1,1.0,0.0001],dtype=torch.float)
# y.backward(v)
# print(x.grad)

x = torch.randn(6,requires_grad=True)
grades = torch.FloatTensor([1,2,3,4,5,6])
y = torch.mean(x**2)
y.backward()
print(x)
print(x.data)
print(x.grad)

