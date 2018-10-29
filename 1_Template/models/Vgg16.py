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

from models.Basicmodule import BasicModule  # 相对导入
import torch.nn as nn

class VGG16(BasicModule):

    def __init__(self, num_classes=2):
        
        super().__init__()
        
        self.model_name = 'vgg16'  # 创建一个名称属性，用于保存model
        
# modify start - based on vggnet in pytorch model zoo
        cfg ={16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
              19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}
        
        self.features = make_layers(cfg[16])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    
    def make_layers(cfg, batch_norm=True):
        layers = []
        in_channels = 3  # 初始通道数为3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)  
  
# modify finish

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

if __name__=='__main__':
    
    net = VGG16(num_classes=2)
    print(net)

