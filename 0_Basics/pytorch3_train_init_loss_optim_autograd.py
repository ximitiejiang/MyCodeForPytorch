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
对于如果使用的pytorch做迁移模型，需要把所有输入图片size转换成224x224，除了inception为299,
需要把图片基先转化到90,1)之间(这一步可以在ToTensor的transforms做)
然后基于mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]进行normalization, 
详细参考：https://pytorch.org/docs/master/torchvision/models.html
-----------------------------------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
import torch.optim as optim

# -----------------1. 定义模型-----------------
input_size = 224  # 适合稍大尺寸图片
model = models.resnet18(pretrained=True)
for param in model.parameters(): # 取出每一个参数tensor
    param.requires_grad = False  # 原始模型的梯度锁定
in_fc = model.fc.in_features
model.fc = nn.Linear(in_fc, 10) # 替换最后一层fc，改为输出为2分类
parameters_to_update = model.fc.parameters()

# -----------------2. 准备数据 (基于CIFAR10进行10分类)-----------------
train_transforms = transforms.Compose([transforms.RandomResizedCrop(input_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                       ])

test_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                      ])
import os
path = '~/MyDatasets/CIFAR10'
fullpath = os.path.expanduser(path)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')
 
trainset = datasets.CIFAR10(root = fullpath,
                            transform = train_transforms,
                            train = True,
                            download = True)

testset = datasets.CIFAR10(root = fullpath,
                           transform = test_transforms,
                           train = False,
                           download = True)
#data, label = trainset[11]  # 数据集输出形式：(data, label)
#print(classes[label])
#print(data.shape)
import matplotlib.pyplot as plt
img = data*0.5 + 0.5
img = np.transpose(img, (1,2,0))
plt.imshow(img)

# -----------------3. 定义训练参数-----------------
batch_size = 4
num_epoch = 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# 使用多块GPU的方法：
#device = torch.devie('cuda:0' if torch.cuda.is_available() else 'cpu')
#if torch.cuda.device_count() > 1:
#    model = nn.DataParallel(model)
#    model.to(device)
optimizer = optim.SGD(parameters_to_update, lr=0.001, momentum=0.9)
loss = nn.CrossEntropyLoss()

trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size = batch_size,
                                          shuffle = True,
                                          num_workers = 2)
testloader = torch.utils.data.DataLoader(testset, 
                                          batch_size = batch_size,
                                          shuffle = True,
                                          num_workers = 2)
# 检查trainloader的输出
len(trainloader)
len(testloader)
for datas, labels in trainloader:
    print(datas)
    print(labels)

# 4. 开始训练
best_model_weights = copy.deepcopy(model.state_dict()) # 把原模型的参数和缓存保存
best_acc = 0.0
since = time.time()

model.train()
for epoch in range(num_epoch):
    print(epoch, '/' , num_epoch)     
    
    for i, inputs in enumarate(trainloader):
        # 数据送入device
        inputs.to(device)
        labels.to(device)
        # 优化器梯度清零
        optimizer.zero_grad()
        # 计算每个batch的输出/损失/预测
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        # 反向传播
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))        

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))   

# 5. 进行验证



'''--------------------------------------------------------
Q. 如何训练一个全新模型？
-----------------------------------------------------------
'''
# 基于自定义resnet18模型
