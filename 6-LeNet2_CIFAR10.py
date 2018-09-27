#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 15:20:49 2018

@author: suliang
"""


import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms import ToPILImage


''' 
创建数据变换器
'''
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5],
                                                     std=[0.5,0.5,0.5])])
''' 
加载数据
'''
trainset = datasets.CIFAR10(root = '/Users/suliang/MyDatasets/CIFAR10/',
                            transform = transform,
                            train = True,
                            download = True)
testset = datasets.CIFAR10(root = '/Users/suliang/MyDatasets/CIFAR10/',
                           transform = transform,
                           train = False,
                           download = True)
# 这是已定义好顺序的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

''' 
分包数据：dataloader函数生成一个可迭代的数据结构, 提供按大小打包，随机分发的服务
'''
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size = 4,
                                          shuffle = True,
                                          num_workers = 2)
testloader = torch.utils.data.DataLoader(testset, 
                                          batch_size = 4,
                                          shuffle = True,
                                          num_workers = 2)

'''
构建神经网络模型
卷积层
(0): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
(1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
(3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  
全联接层
(0): Linear(in_features=400, out_features=120, bias=True)
(1): Linear(in_features=120, out_features=84, bias=True)
(2): Linear(in_features=84, out_features=10, bias=True)

'''
import torch
import torch.nn as nn
 
class LeNet(nn.Module):
 
    def __init__(self):
        #Net继承nn.Module类，这里初始化调用Module中的一些方法和属性
        nn.Module.__init__(self) 
 
        #定义特征工程网络层，用于从输入数据中进行抽象提取特征
        self.feature_engineering = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=6,
                      kernel_size=5),
 
 
            #kernel_size=2, stride=2，正好可以将图片长宽尺寸缩小为原来的一半
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
 
            nn.Conv2d(in_channels=6,
                      out_channels=16,
                      kernel_size=5),
 
 
            nn.MaxPool2d(kernel_size=2,
                        stride=2)
        )
 
        #分类器层，将self.feature_engineering中的输出的数据进行拟合
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16*5*5,
                      out_features=120),
 
 
            nn.Linear(in_features=120,
                      out_features=84),
 
 
            nn.Linear(in_features=84,
                      out_features=10),
 
        )

net = LeNet()
print(net) 
 
 
    def forward(self, x):
        #在Net中改写nn.Module中的forward方法。
        #这里定义的forward不是调用，我们可以理解成数据流的方向，给net输入数据inpput会按照forward提示的流程进行处理和操作并输出数据
        x = self.feature_engineering(x)
        x = x.view(-1, 16*5*5)
        x = self.classifier(x)
        return x
