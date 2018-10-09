#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 09:38:56 2018

@author: suliang
"""

import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

# 定义数据转换器：把每张图片的2维数据转成一维tensor，把tensor归一化
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5],
                                                     std=[0.5,0.5,0.5])])
# 导入数据: 会自动生成2个文件夹，一个文件夹raw放置下载的原始数据文件并自动解压缩，
# 另一个文件夹processed放置处理后的文件training.pt, test.pt
data_train = datasets.MNIST(root = '/Users/suliang/MyDatasets/MNIST/',
                            transform = transform,
                            train = True,
                            download = False)
data_test = datasets.MNIST(root = '/Users/suliang/MyDatasets/MNIST/',
                           transform = transform,
                           train = False)

# 对数据进行分包成batch,设置shuffle=true则会随机输出batch
data_loader_train = torch.utils.data.DataLoader(dataset = data_train,
                                                batch_size = 64,
                                                shuffle = True)
data_loader_test = torch.utils.data.DataLoader(dataset = data_test,
                                               batch_size = 64,
                                               shuffle = True)
# 做可视化
import matplotlib.pyplot as plt
images, labels = next(iter(data_loader_train))  # iter()生成一个迭代对象，next()返回每个值
img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1,2,0)
std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img*std+mean   # 数据预处理做了归一化，所以这里做归一化的逆运算还原图片

print([labels[i] for i in range(64)])
plt.imshow(img)

# 搭建模型
class Model(torch.nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        # 卷积层：卷积核3x3, 步长1，padding填充1层
        # 最大池化层
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,64,kernel_size=3,stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))
        
        self.dense = torch.nn.Sequential(torch.nn.Linear(14*14*128,1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024,10))
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x


# 创建模型对象
model = Model()
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())  # lr使用了默认学习率

print(model)
        
# 模型训练
epoch_n = 5
for epoch in range(epoch_n):
    running_loss=0
    running_correct = 0
    print('epoch {}/{}:'.format(epoch, epoch_n))
    print('-'*20)
    
    for data in data_loader_train:
        x_train, y_train = data
        x_train, y_train = Variable(x_train), Variable(y_train)
        outputs = model(x_train)
        _,pred = torch.max(outputs.data, 1)
        
        optimizer.zero_grad()
        loss = cost(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data[0]
        
   
    y_pred = models(x)   # 
    
    loss = loss_fn(y_pred, y)
    print('Epoch: {}, Loss:{:.4f}'.format(epoch, loss.data[0]))
    optimizer.zero_grad()  # 对模型参数梯度归零
    
    loss.backward()
    
    optimizer.step()        
        
        
        
        
