#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:23:30 2018

@author: ubuntu
"""

'''--------------------------------------------------------
Q. 如何训练一个简单的迁移模型？
对于如果使用的pytorch做迁移模型，需要把所有输入图片size转换成224x224，除了inception为299,
需要把图片基先转化到90,1)之间(这一步可以在ToTensor的transforms做)
然后基于mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]进行normalization, 
详细参考：https://pytorch.org/docs/master/torchvision/models.html

1. 时间分析：
# 数据集CIFAR10：一共60000张图片（训练图片50000张，测试图片10000张），图片大小32x32，转化为3x224x224
# 模型resnet18迁移模型，只训练最后一层fc的参数
# 损失函数：交叉熵, nn.CrossEntropyLoss()
# 学习率：0.001, optimizer = optim.SGD(parameters_to_update, lr=0.001, momentum=0.9)
# 1 epoch on CPU: 38m 42s
# 1 epoch on GPUx2: 4m 48s, 48m(for 10 epochs) 

2. 固定学习率跟指数衰减学习率，对训练loss和验证loss的影响？
* 固定学习率
1 / 10
Loss: 2.0178 Acc: 0.3321
2 / 10
Loss: 1.9719 Acc: 0.3593
3 / 10
Loss: 1.9619 Acc: 0.3647
4 / 10
Loss: 1.9547 Acc: 0.3673
5 / 10
Loss: 1.9542 Acc: 0.3681
6 / 10
Loss: 1.9522 Acc: 0.3699
7 / 10
Loss: 1.9528 Acc: 0.3682
8 / 10
Loss: 1.9518 Acc: 0.3703
9 / 10
Loss: 1.9603 Acc: 0.3692
10 / 10
Loss: 1.9558 Acc: 0.3673
Training complete in 48m 0s
* 指数衰减学习率



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
model = models.resnet18(pretrained=True)
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

# CIFAR10：一共60000张图片（训练图片50000张，测试图片10000张），图片大小32x32
#          一共10个分类，每个分类6000张
trainset = datasets.CIFAR10(root = fullpath,
                            transform = train_transforms,
                            train = True,
                            download = True)

testset = datasets.CIFAR10(root = fullpath,
                           transform = test_transforms,
                           train = False,
                           download = True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')
print('Length of trainset:{}'.format(len(trainset)))

#data, label = trainset[11]  # trainset输出形式：list(data, label),所以可以切片
#print(classes[label])
#print(data.shape)

#img = data*0.5 + 0.5
#img = np.transpose(img, (1,2,0)) # 还有别的维度顺序转换方法吗，np.transpose不像pytorch的写法
#plt.imshow(img)

# -----------------3. 定义训练参数-----------------
batch_size = 4
num_epoch = 4
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
print('Length of trainloader:{}'.format(len(trainloader)))

# -----------------4. 开始训练-----------------
print('Start training...')

since = time.time()

# 是否需要设置这步model.train()
#model.train()

from tqdm import tqdm
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
# 其中VisdomPlotLogger用于plot各种线条
#     VisdomLogger用于绘制confusion matrix

loss_meter = tnt.meter.AverageValueMeter()
confusion_matrix = tnt.meter.ConfusionMeter(10)
import visdom

train_acc_hist = []   
train_loss_hist = []
for epoch in range(num_epoch):
    print(epoch+1, '/' , num_epoch)     
    
    loss_meter.reset()
    confusion_matrix.reset()
    
    
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in tqdm(trainloader): # dataloader输出形式：iter(data, labels),所以可以迭代
        # 数据转换为torch.cuda.FloatTensor
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 优化器梯度清零
        optimizer.zero_grad()
        # 计算每个batch的输出/损失/预测
        outputs = model(inputs)  
        loss = criterion(outputs, labels) # 计算的是1个batch4张图的平均损失
        _, preds = torch.max(outputs, 1)
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 损失计算：
        # 1.loss
        # 2.running loss
        # 3.epoch loss
        running_loss += loss.item() * inputs.size(0)   # 计算4张图的总损失，并累加
        running_corrects += torch.sum(preds == labels.data)
        
        
        loss_meter.add(loss.item())
        confusion_matrix.add(outputs.detach(), labels.detach())
    
    epoch_loss = running_loss / len(trainloader.dataset)  # 计算1个epoch下，平均每张图的损失
    epoch_acc = running_corrects.double() / len(trainloader.dataset)
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))  
    train_acc_hist.append(epoch_acc)    
    train_loss_hist.append(epoch_loss)
    
    loss_logger = VisdomPlotLogger('line', win='loss', opts={'title':'Loss'}, port=8097, server='localhost')
    acc_logger = VisdomLogger('heatmap', win='acc', opts={'title': 'Confusion matrix','columnnames': list(range(10)),'rownames': list(range(10))}, port=8097, server='localhost')
    
    #train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    #confusion_logger.log(confusion_meter.value())
    
    loss_logger.log(epoch,loss_meter.value()[0])
    acc_logger.log(confusion_matrix.value())


time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# -----------------5. 训练可视化-----------------

ahist = []
lhist = []

ahist = [h.cpu().numpy() for h in train_acc_hist]  # train_acc_hist为list[tensor]
lhist = [h for h in train_loss_hist]               # train_loss_hist为list[float]

plt.title("Loss")
plt.xlabel("Training Epochs")
plt.ylabel("Loss")
plt.plot(range(1,num_epoch+1), lhist,label="Pretrained")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epoch+1, 1.0))
plt.legend()
plt.show()   


plt.title("Accuracy")
plt.xlabel("Training Epochs")
plt.ylabel("Acc")
plt.plot(range(1,num_epoch+1), ahist,label="Pretrained")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epoch+1, 1.0))
plt.legend()
plt.show()   

# -----------------5. 验证及训练可视化-----------------
print('Start validating...')

best_model_weights = copy.deepcopy(model.state_dict()) # 把原模型的参数和缓存保存
best_acc = 0.0

val_acc_hist = []
val_loss_hist = []

