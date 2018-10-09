#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:18:24 2018

@author: suliang
"""

'''
对比手动求导和pytorch自动求导器的结果

'''
import torch
from torch.autograd import Variable

# 手动求导
x = torch.randn(3,4)
y = x**2 * x.exp()
hand_grad = 2*x*x.exp() + x**2*x.exp()
print(hand_grad)


# 自动求导
x = Variable(x, requires_grad=True)
y = x**2 * x.exp()
y.backward(torch.ones(y.shape))    
# 默认backward只支持标量tensor，而对矢量tensor需要定义backward(grad_variables)参数
# 令grad_variables=全1矩阵，大小跟根结点shape相同，也就是=1就求导，所以全=1就全求导
print(x.grad)

