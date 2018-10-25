#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 08:22:36 2018

@author: ubuntu

1. class中变量的来源：
    - 可以从init函数的input形参，然后在init里边用self.xx转化为类全局变量
    - 可以在任意函数形参输入，只用于这个函数内做局部变量
    - 可以在任意函数内定义新变量，用于本函数，或self.xx转化为类全局变量

2. 类继承，可以获得原来类的方法，比如.load(), .save()
   类继承，需要先导入原始类，比如导入nn.Module，就要先导入torch.nn
                            比如导入BasicModule，就要先导入BasicModule

"""

from .Basicmodule import BasicModule  # 相对导入
import torch.nn as nn

class AlexNet(BasicModule):

    def __init__(self, num_classes=2):
        
        super(AlexNet, self).__init__()
        
        self.model_name = 'alexnet'  # 创建一个名称属性，用于保存model
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

if __name__=='__main__':
    
    anet = AlexNet(num_classes=4)
    print(anet)
    print(anet.model_name)  # 模型加载，输出模型名字
    anet.save()            # 从basicmodule继承的save()函数
