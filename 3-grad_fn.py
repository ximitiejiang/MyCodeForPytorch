#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:18:24 2018

@author: suliang
"""

'''
查看grad前进步骤
- z.grad_fn                  查看z的梯度函数
- z.grad_fn.next_functins    查看z下一级的梯度函数
- z.grad_fn.saved_variables  查看

'''
import torch
from torch.autograd import Variable

# 定义输入，和参数
x = Variable(torch.ones(1))
w = Variable(torch.rand(1), requires_grad = True)
b = Variable(torch.rand(1), requires_grad = True)
# 定义前向计算过程
y = w * x
z = y + b

print(x.requires_grad, w.requires_grad, b.requires_grad, y.requires_grad, z.requires_grad)
print(x.is_leaf, w.is_leaf, b.is_leaf)

print(z.grad_fn)            # z自己的梯度函数是add的反向传播，所以是addBackward函数
print(z.grad_fn.next_functions)  # z之后的梯度函数包括y自己的乘法反传，以及b自己的梯度累加

print(y.grad_fn)            # y自己的梯度函数是乘法的反向传播，所以是mulBackward函数
print(y.grad_fn.next_functions)  # y之后的梯度函数包括w的梯度累加，x没有梯度累加所以为none

print(w.grad_fn)   # w自己没有反向传播函数，虽然有梯度积累但显示为none

# 通常一组输入计算一轮，会有一次backward反向传播就够了，所以系统默认一次反向传播后就清空buffer
# 每次backward计算完，就会释放整个计算图buffer，如果再执行backward就会因没有变量而报错
# 因此在写代码时，如果一次数据输入需要两次backward就需要设置backward(retain_graph=True)
z.backward(retain_graph =True)  
print(w.grad)
print(x)
print(y)
print(y.grad)
