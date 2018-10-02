#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 16:42:30 2018

@author: suliang

迁移学习实例：基于imagenet的一个子集，包含bees和ant两个类

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

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    
    # 储存最优模型的参数（深拷贝，不会相互影响）
    best_model_wts = copy.deepcopy(model.state_dict())  # 拷贝模型的状态字典state_dict
    best_acc = 0.0         # 最优模型

    for epoch in range(num_epochs):  # 外循环：epoch个数
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:  # 中循环： 训练集和测试集
            # 训练集，则指定？？？
            if phase == 'train':
                scheduler.step()  # ？？？
                model.train()  # ??? Set model to training mode
            # 测试集，则？？？
            else:
                model.eval()   # ???? Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:  # 内循环： 每个batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 每个batch作为最小训练单元，都需要清空上一次训练的梯度
                optimizer.zero_grad()

                # forward
                # track history if only in train
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
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

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
        

'''
建立模型和设置模型
'''    
#import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)   # 下载完成
#alexnet = models.alexnet(pretrained=True)    # 下载完成
#squeezenet = models.squeezenet1_0(pretrained=True)
#vgg16 = models.vgg16(pretrained=True)
#densenet = models.densenet161(pretrained=True)
#inception = models.inception_v3(pretrained=True)

# 建立迁移学习resnet18的模型和参数
model_ft = models.resnet18(pretrained=True)

print(resnet18)
# 设置
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
# 设置运行设备
model_ft = model_ft.to(device)
# 设置损失函数
criterion = nn.CrossEntropyLoss()
# 设置优化器
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# 设置学习率
# 每隔7个step就把学习率变为原来的gamma倍Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)    


'''
训练模型
'''
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)


'''
可视化模型输出
'''
visualize_model(model_ft)