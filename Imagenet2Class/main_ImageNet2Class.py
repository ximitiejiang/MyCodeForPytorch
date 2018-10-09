#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:42:30 2018

@author: suliang

迁移学习实例：基于imagenet的一个子集，包含bees和ant两个类
基于pytorch英文网站实例
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode

'''
导入数据，定义数据变换
- 随机切割
- 随机翻转
- 张量转换(归一化)
- 标准化(pytorch官方推荐的mean, std, 不知道是不是比[0.5,0.5,0.5]更好)
注意： 验证集不做数据扩增，只做基础处理resize/centercrop/totensor/normalize
'''
# 定义transforms，同时定义train和val
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/Users/suliang/MyDatasets/Imagenet_2class/hymenoptera_data'
# 读取数据: 通过imageFolder同时读取train和val
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
# 分包数据: 同时分包train和val
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
# 设备指定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''
单个图片可视化
该函数较通用，实现tensor的显示(逆归一化+图片和标签打印)
'''
def imshow(inp, title=None):  
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0)) # 图片转成numpy, 并由C,H,W变成H,W C
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean   # 逆归一化
    inp = np.clip(inp, 0, 1)  # 切割数据在0,1之间
    plt.imshow(inp)
    if title is not None:   # 用于打印对应的类别标签
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# 获得一个batch
inputs, classes = next(iter(dataloaders['train']))
# 拼接整个batch
out = torchvision.utils.make_grid(inputs)
# 这个函数比plt.imshow()好用些
imshow(out, title=[class_names[x] for x in classes])


'''
函数：训练模型

'''
# 传入模型，优化器，学习率规划器
def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
    since = time.time()
    
    
    # 储存最优模型的参数（深拷贝，不会相互影响）
    best_model_wts = copy.deepcopy(model.state_dict())  # 拷贝模型的状态字典state_dict
    best_acc = 0.0         # 最优模型

    for epoch in range(num_epochs):  # 外循环：epoch个数
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  # 中循环： 训练模式和验证模式
            # 训练模式设定
            if phase == 'train':
                scheduler.step()  # 训练模式：规划器定义指数损失学习率
                model.train()     # 设定模型为训练模式（drop层工作）
            # 测试模式
            else:
                model.eval()      # 设定模型为验证模式（drop层不工作）

            running_loss = 0.0
            running_corrects = 0

            # 内循环：每个batch(4张图片为1个batch)
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 每个batch作为最小训练单元，都需要清空上一次训练的梯度
                optimizer.zero_grad()

                # 训练模式下：计算前向输出，预测，损失
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算loss, 正确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 如果新训练出的模型参数比原有最优模型参数更好，则更新
            # 并且深拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

'''
函数：可视化模型
'''
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()  # 评估模式
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():  # 关闭梯度计算
        # 取出验证集的数据
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 计算模型的输出和预测值
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
        
if __name__ == '__main__':
    
    method = 2
    
    if method == 1:
        '''
        第一种训练模型的方式：迁移模型，所有参数重新训练----------------------------
        在2个epoch下耗时2m 60s
        '''    
        # 建立迁移学习resnet18的模型和参数
        model_ft = models.resnet18(pretrained=True)
        # 重新定义全联接层  这种调用方式很特别：model的层，层的参数，都可以通过.xx调用
        num_ftrs = model_ft.fc.in_features   
        model_ft.fc = nn.Linear(num_ftrs, 2) # 重新定义全联接层，输出为2个节点，即2分类
        # 设置运行设备
        model_ft = model_ft.to(device)
        # 设置损失函数
        criterion = nn.CrossEntropyLoss()
        # 设置优化器
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        # 设置学习率：指数衰减学习率
        # 每隔7个step就把学习率变为原来的gamma倍Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)    
        '''
        训练模型
        '''
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=2)
        
        '''
        可视化模型输出
        '''
        # 训练完成后，输出训练结果
        visualize_model(model_ft)
        
    elif method == 2:
        '''
        另外一种训练方式：迁移模型，只训练最后一层
        在2个epoch下耗时1m 27s
        第二种方法耗时比第一种少了一半以上，但精度上可能需要更多的epoch来做。
        '''
        # 先冻结所有参数
        model_conv = torchvision.models.resnet18(pretrained=True)
        for param in model_conv.parameters():
            param.requires_grad = False
            
            # 新增加fc层，默认求导
            num_ftrs = model_conv.fc.in_features
            model_conv.fc = nn.Linear(num_ftrs, 2)
            
            model_conv = model_conv.to(device)
            
            criterion = nn.CrossEntropyLoss()
            
            # 只把fc层的参数传递给优化器进行优化
            optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
            
            # 同样采用指数衰减优化方式
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
            
            # 开始训练模型
            model_conv = train_model(model_conv, criterion, optimizer_conv,
                                     exp_lr_scheduler, num_epochs=4)
            
            visualize_model(model_conv)
            
            plt.ioff()
            plt.show()
    
    else:
        print('Wrong test_id!')
    

