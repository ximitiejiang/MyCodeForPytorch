#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:11:48 2018

@author: ubuntu
"""

'''--------------------------------------------------------
Q. 如何训练一个简单的迁移模型？
-----------------------------------------------------------
'''
import torch.nn as nn
from torchvision import models
# 0. 准备数据


# 1. 初始参数定义
batch_size = 4
num_epoch = 10
input_size = 224

# 2. 定义模型
model = models.resnet18(pretained=True)
in_fc = model.fc.in_features
model.fc = nn.Linear(in_fc, 2)

# 2. 模型初始化
def initialize_model():
    model = models.resnet18(pretrained = True)
    model.fc = nn.Linear(512,10)
    
model = initialize_model()

# 3. 




'''--------------------------------------------------------
Q. 如何训练一个全新模型？
-----------------------------------------------------------
'''
# 基于自定义resnet18模型
