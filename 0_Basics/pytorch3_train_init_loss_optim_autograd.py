#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:11:48 2018

@author: ubuntu
"""


'''--------------------------------------------------------
Q. 如何定义优化器optimizer？
-----------------------------------------------------------
'''
optimizer = 


'''--------------------------------------------------------
Q. 如何定义损失函数criteria？
-----------------------------------------------------------
'''
criteria = 


'''--------------------------------------------------------
Q. 如何训练一个简单的迁移模型？
-----------------------------------------------------------
'''
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms

# 1. 定义模型
input_size = 224
model = models.resnet18(pretrained=True)
for param in model.parameters(): # 取出每一个参数tensor
    param.requires_grad = False  # 原始模型的梯度锁定
in_fc = model.fc.in_features
model.fc = nn.Linear(in_fc, 10) # 替换最后一层fc，改为输出为10分类
parameters_to_update = model.fc.parameters()

# 2. 准备数据 (基于CIFAR10进行10分类)
train_transforms = transforms.Compose([transforms.RandomResizedCrop(input_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])

test_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])
import os
path = '~/MyDatasets/CIFAR10/'
fullpath = os.path.expanduser(path)
   
trainset = datasets.CIFAR10(root = fullpath,
                            transform = train_transforms,
                            train = True,
                            download = True)

testset = datasets.CIFAR10(root = fullpath,
                           transform = test_transforms,
                           train = False,
                           download = True)

trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size = 4,
                                          shuffle = True,
                                          num_workers = 2)

testloader = torch.utils.data.DataLoader(testset, 
                                          batch_size = 4,
                                          shuffle = True,
                                          num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# 3. 定义训练参数
batch_size = 4
num_epoch = 20
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# 使用多块GPU的方法：
#device = torch.devie('cuda:0' if torch.cuda.is_available() else 'cpu')
#if torch.cuda.device_count() > 1:
#    model = nn.DataParallel(model)
#    model.to(device)
optimizer = optim.SGD(parameters_to_update, lr=0.001, momentum=0.9)
criteria = nn.CrossEntropyLoss()

# 4. 开始训练
best_model_weights = copy.deepcopy(model.state_dict()) # 把原模型的参数和缓存保存
best_acc = 0.0
since = time.time()

for epoch in num_epoch:        
        for data, label in trainloader:
            
        

# 5. 进行验证



'''--------------------------------------------------------
Q. 如何训练一个全新模型？
-----------------------------------------------------------
'''
# 基于自定义resnet18模型
