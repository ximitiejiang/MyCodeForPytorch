#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 13:51:15 2018

@author: suliang
"""
import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

to_tensor = ToTensor()
to_pil = ToPILImage()

lena = Image.open('lena.jpg')

input = to_tensor(lena)
input = input.view(1,3,300,300)  # 对所有layer须输入batch数据，这里升维就是定义batch(B,C,H,W)

kernel = torch.ones(3,3)/-9.0
kernel[1][1] = 1

conv = torch.nn.Conv2d(3,3,(3,3),stride=1,bias=False) 
list(conv.parameters())
# 参数：(in_channel, out_channel, kernel_size, stride, bias=True)
conv.weight.data = kernel.view(3,3,3,3)
out = conv(Variable(input))

to_pil(out.data.squeeze(0))