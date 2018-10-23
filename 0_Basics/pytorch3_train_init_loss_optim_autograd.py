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
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


'''--------------------------------------------------------
Q. 如何定义损失函数criteria？
1.损失函数是什么：用来表示预测值与标签值之间的偏差的函数，对损失函数进行梯度计算来得到预测函数的参数。
  为了能够求出参数，损失函数的设计要考虑设计成便于求梯度，有极大值的凸函数。
2.常见机器学习损失函数：
    - LiR/线性回归：欧式距离，等于min欧式距离，最后欧式距离求极值转化为梯度计算求参数
    - LoR逻辑回归：最大似然估计，等于最大似然函数即max联合概率，取log连乘变连加，最后最大似然函数求极值转化为梯度计算求参数
    - Softmax回归：交叉熵，等于min交叉熵，最后交叉熵求极值转化为梯度计算求参数
    - SVM分类：
    - 决策树分类：损失函数是极大似然，但实现时是用最优特征选择，即信息增益或基尼指数，即递归方式通过最大信息增益或者最小基尼指数选择最优特征。
3.深度学习损失函数：
    - 均方误差损失函数torch.nn.MSELoss
    - 负log似然损失函数torch.nn.NLLLoss
    - 交叉熵损失函数torch.nn.CrossEntropyLoss
-----------------------------------------------------------
'''
# MSE损失函数：即均方误差损失函数
criterion = nn.MSELoss()
# NLLLoss损失函数：即负log似然损失，用于多分类问题，需要配合最后添加一层log softmax layer
criterion = nn.NLLLoss()
# 交叉熵损失函数：整合nn.LogSoftmax()和nn.NLLLoss()在一个class，所以不需要配合添加log softmax layer
criterion = nn.CrossEntropyLoss()


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

import time
import os
import copy
# -----------------1. 定义模型-----------------
input_size = 224  # 适合稍大尺寸图片
model = models.resnet34(pretrained=True)
for param in model.parameters(): # 取出每一个参数tensor
    param.requires_grad = False  # 原始模型的梯度锁定
in_fc = model.fc.in_features
model.fc = nn.Linear(in_fc, 10) # 替换最后一层fc，改为输出为10分类
parameters_to_update = model.fc.parameters()

# -----------------2. 准备数据 (基于CIFAR10进行10分类)-----------------
# 需要注意datasets和dataloader的作用和输出形式的区别：
# datasets是导入数据，输出形式是list，可切片
# dataloader是分包数据，输出形式是iter，可迭代
train_transforms = transforms.Compose([transforms.RandomResizedCrop(input_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                       ])

test_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                                      ])

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
data, label = trainset[11]  # datasets输出形式：list(data, label),所以可以切片
print(classes[label])
print(data.shape)

img = data*0.5 + 0.5
img = np.transpose(img, (1,2,0)) # 还有别的维度顺序转换方法吗，np.transpose不像pytorch的写法
plt.imshow(img)

# -----------------3. 定义训练参数-----------------
batch_size = 4
num_epoch = 1
#device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #cuda起始编号为0,而不是第0个cuda
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model,device_ids=[0, 1]) # 如果不写ids则默认使用全部GPU，需注意编号与前句device对应
model.to(device)
    
optimizer = optim.SGD(parameters_to_update, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size = batch_size,
                                          shuffle = True,
                                          num_workers = 2)
testloader = torch.utils.data.DataLoader(testset, 
                                          batch_size = batch_size,
                                          shuffle = True,
                                          num_workers = 2)

# -----------------4. 开始训练-----------------
best_model_weights = copy.deepcopy(model.state_dict()) # 把原模型的参数和缓存保存
best_acc = 0.0
since = time.time()

# 是否需要设置这步model.train()
#model.train()

for epoch in range(num_epoch):
    print(epoch+1, '/' , num_epoch)     
    
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in trainloader: # dataloader输出形式：iter(data, labels),所以可以迭代
        # 数据转换为torch.cuda.FloatTensor
        inputs = inputs.to(device)
        labels = labels.to(device)
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
    
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = running_corrects.double() / len(trainloader.dataset)
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))        

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# -----------------5. 验证及可视化-----------------


'''--------------------------------------------------------
Q. 如何训练一个全新模型？
- 从头写一个Resnet34, 并训练模型，看性能跟迁移模型差别多少
-----------------------------------------------------------
'''
import torch
import torch.nn as nn

# -----------------1. 定义模型-----------------
class SimpleResnet34(nn.Module):
    
# -----------------2. 准备数据 (基于CIFAR10进行10分类)-----------------

# -----------------3. 定义训练参数-----------------

# -----------------4. 开始训练-----------------

# -----------------5. 验证及可视化-----------------