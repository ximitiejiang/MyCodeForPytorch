#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 09:10:31 2018

@author: ubuntu
"""

import torch
from torch import nn

class Linear(nn.Module):
    '''创建linear层，继承自nn.module
    linear参数w,b先通过nn.Parameter()类的__new__()方法生成参数
    '''
    def __init__(self, in_features, out_features):
        super().__init__()
    
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        x = x.mm(self.w)   # 乘法：代表x*w
        return x + self.b.expand_as(x)


layer = Linear(4,3)
input = torch.randn(2,4)
output = layer(input)

print(output)
        