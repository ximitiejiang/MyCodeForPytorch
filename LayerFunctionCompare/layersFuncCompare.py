#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 13:51:15 2018

@author: suliang
"""
import torch
from torchvision import datasets, models, transforms
from PIL import Image
#from torchvision.transforms import ToTensor, ToPILImage

'''
定义变换器
'''
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

'''
载入图片，并转换成tensor/batch
'''
lena = Image.open('lena.jpg')
print(lena.size)                 # 图片本体, 只能用.size, 显示的是图片像素HxW
data = to_tensor(lena)
print(data.size())               # 图片tensor, 需要用.size(), 显示的是图片CxHxW
data = data.unsqueeze(0)         # 图片batch, 显示的是图片batch(B,C,H,W)
print(data.size()) 

'''
定义卷积核
'''
kernel = torch.ones(3,3)/-9.0    
kernel[1][1] = 1
print(kernel)

# 定义卷积层
conv = torch.nn.Conv2d(3,3,(3,3),stride=1,bias=False) 
list(conv.parameters())
# 参数：(in_channel, out_channel, kernel_size, stride, bias=True)

conv.weight.data = kernel.view(3,3,3,3)
out = conv(data)

img = to_pil(out.data.squeeze(0))