#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 09:49:10 2018

@author: suliang

参考‘深度学习框架pytorch入门与实践’ chapter-2, 正确率53%
该代码来自于pytorch官方英文教程，实现LeNet5

"""

import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, datasets
from torchvision.transforms import ToPILImage
import time

''' 
创建数据变换器
'''
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5],
                                                     std=[0.5,0.5,0.5])])
''' 
加载数据
'''
trainset = datasets.CIFAR10(root = '/Users/suliang/MyDatasets/CIFAR10/',
                            transform = transform,
                            train = True,
                            download = True)
testset = datasets.CIFAR10(root = '/Users/suliang/MyDatasets/CIFAR10/',
                           transform = transform,
                           train = False,
                           download = True)
# 数据加载后直接使用：
# trainset是一个可迭代对象，等效于set(data, label)
data, label = trainset[0]

# 这是已定义好顺序的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

''' 
查看trainset中的第102张图片内容
'''
data, label = trainset[102]    # tensor, size() = 3x32x32
print(classes[label])
data = data*0.5 + 0.5  # 归一化的逆运算
#img = ToPILImage(data)  #不知道为什么这个命令不能显示
#img.show()
import matplotlib.pyplot as plt
import numpy as np
data = np.transpose(data, (1,2,0))  # 转秩，从CxHxW变为HxWxC, shape = 32x32x3
plt.imshow(data)  # plt.imshow()只能显示numpy的HxWxC格式

''' 
分包数据：dataloader函数生成一个可迭代的数据结构, 提供按大小打包，随机分发的服务
'''
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size = 4,
                                          shuffle = True,
                                          num_workers = 2)
testloader = torch.utils.data.DataLoader(testset, 
                                          batch_size = 4,
                                          shuffle = True,
                                          num_workers = 2)
''' 
查看第一个随机batch的图片
'''
data, label = next(iter(trainloader))   # size = 4x3x32x32
data = data*0.5 + 0.5
# make_grid可以把一个batch的数据重组拼接成一行，所以batch=4的图片被排列为1行
# 变为3x32x128, 同时默认有p=2的padding，所以变为3x(32+4)x(128+10)
data = torchvision.utils.make_grid(data)  # size = 3x36x138
# data = T.resize(200)(data)  # 图片变大，同样显示空间像素更多了，看会不会变清楚？
data = np.transpose(data, (1,2,0))
plt.imshow(data)

'''
构建网络
- 需要继承父类网络，并定义每一层网络类型和节点数
- 需要定义前向计算
'''
import torch
import torch.nn.functional as F

class LeNet1(torch.nn.Module):
    def __init__(self):
        super(LeNet1, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(3,6,5)  # 输入3通道，输出6通道，卷积核5x5
        self.conv2 = torch.nn.Conv2d(6,16,5) # 输入6通道，输出16通道，卷积核5x5
        
        self.fc1 = torch.nn.Linear(16*5*5, 120) #
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'''
新建网络对象
'''

# 初始化   
net = LeNet1()
criteria = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)


'''
打印模型参数进行查看
'''
f = open("parmas.txt", "w") 
print(net, file = f)
print('-'*20,file=f)

# 查看model.parameters()
params = list(net.parameters())  # 获得该网络所有可学习参数列表 - 已初始化，未训练
print('length of params: '.format(len(params))) # 打印参数长度 = 10，params[0]就代表conv1的参数

# 查看model.state_dict()
print("Model's state_dict:", file = f)
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size(), file=f)
print('-'*20,file=f)
# 查看optimizer.state_dict()
print("Optimizer's state_dict:", file = f)
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name], file=f)
print('-'*20,file=f)
f.close()


'''
训练网络
'''
since = time.time()
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0): # 取出每个batch
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criteria(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss +=loss.data[0]
        if i%2000 == 1999:
            print('[%d, %5d] loss: %.3f' %(epoch+1, i+1, running_loss/2000))
            running_loss = 0.0
print('finished training!')
print('last time: {}'.format(time.time()-since))
print()

# 查看model.state_dict()
f = open("parmas.txt", "a+") # 只有模式a和a+能够让指针在文件末尾

print("After training, Model's state_dict:", file = f)
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size(), file=f)
print('-'*20,file=f)
# 查看optimizer.state_dict()
print("After training, Optimizer's state_dict:", file = f)
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name], file=f)
print('-'*20,file=f)
f.close()

'''
保存模型
'''
PATH = '/Users/suliang/MyCodeForPytorch/CIFAR10_LeNet5/saved_model'
torch.save(net.state_dict(), PATH)

'''
预测1个batch看看效果
'''
images, labels = next(iter(testloader))
for i in range(4):
    print(classes[labels[i]])
newimgs = images*0.5 + 0.5
newimgs = torchvision.utils.make_grid(newimgs)
newimgs = np.transpose(newimgs,(1,2,0))
plt.imshow(newimgs)

# 每一个image会得到一个概率数组，包含对每个类预测概率值，其中概率最高值就是预测
outputs = net(Variable(images))    
_,predicted = torch.max(outputs.data, 1)  # 返回output中每行最大值，1代表列坍塌
for i in range(4):
    print(classes[predicted[i]])

'''
预测整个数据集
'''
correct = 0 # 初始化正确张数 = 0
total = 0   # 初始化总张数 = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('correct rate: {}%'.format(100*correct/total))  
# tensor相除，小于0的结果会显示成0，所以乘以100就是百分比的显示方式


'''
检测是否可以在GPU运行
'''
if torch.cuda.is_available():
    net.cuda()
    images = images.cuda()
    labels = labels.cuda()
    outputs = net(variable(images))
    loss = criteria(outputs, Variable(labels))
    

