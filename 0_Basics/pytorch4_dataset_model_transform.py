#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:13:21 2018

@author: ubuntu
"""


'''--------------------------------------------------------
Q. 如何定义对图片预处理的transform？
-----------------------------------------------------------
'''
import os
path = '~/MyDatasets/DogsVSCats/train/dog.7014.jpg'  # 待获得完整路径的目录名要以～开头
fullpath = os.path.expanduser(path)   # 获得完整路径名
print(fullpath)

# 先显示原始图片
from PIL import Image
import matplotlib.pyplot as plt
root = '/Home/MyDatasets/DogsVSCats/train/dog.7014.jpg'
data = Image.open(root)
plt.imshow(data)

from torchvision import transforms

# transform1: 缩放和改尺寸, resize(等效于scale, scale已经废弃用resize替代了)
transform1 = transforms.Compose([transforms.Resize([64,64])])  # 改为指定尺寸HxW，如果1个数则只改短边但保持长宽比不变
new_data = transform1(data)
plt.imshow(new_data)

# transform2: 切割, CenterCrop
transform2_1 = transforms.Compose([transforms.CenterCrop((224,400))])  # 基于中心点切出HxW图片
new_data = transform2_1(data)
plt.imshow(new_data)
transform2_2 = transforms.Compose([transforms.RandomResizedCrop(224)]) # 随机切，然后再扩展成size尺寸
new_data = transform2_2(data)
plt.imshow(new_data)

# transform3: 翻转, RandomHorizontalFlip
transform3 = transforms.Compose([transforms.RandomHorizontalFlip()])  # 随机（0.5的概率）水平翻转
new_data = transform3(data)
plt.imshow(new_data)

# transform4: 变为张量，ToTensor
# - 一方面把图片HxWxC，转换为张量的CxHxW
# - 另一方面把图片(0,255)，转换为张量(0,1)
transform4_1 = transforms.Compose([transforms.ToTensor()])  # 图像转成tensor
PtoT = transform4(data)
print('tensor size: {}'.format(PtoT.shape))
print(PtoT.min(), PtoT.max())

transform4_2 = transforms.Compose([transforms.ToPILImage()])  # tensor转成图像- 也可直接用transpose
TtoP = transform4_2(PtoT)
plt.imshow(TtoP)

TtoP = np.transpose(PtoT, (1,2,0))  # ToPILImage所做的事情跟transpose是一样的
plt.imshow(TtoP)

# transform5: 归一化，一般归一化到(-1,1), 因为
# new_value = (value - mean)/std, 后续恢复图片也要进行归一化的逆操作
# 归一化只针对tensor，所以ToTensor与Normalize一般一起做。
# 由于totensor已经把value转换到(0,1), 为了标准化到(-1,1)需要合适的均值/方差
# 简化处理是设置mean=0.5, std=0.5，即(value-0.5)/0.5得到(-1,1)的区间
transform5 = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])])
new_data = transform5(data)
print('tensor size: {}'.format(new_data.shape))
print(new_data.min(), new_data.max())
print('mean: {}, std: {}'.format(new_data.mean(), new_data.std()))

# 标准化以后的数据转换回来再显示
new_data = new_data*0.5 + 0.5
#TtoP = np.transpose(new_data, (1,2,0)) 
TtoP = transforms.ToPILImage()(new_data) # 注意这种蛋疼写法，T.xxx要么嵌套在compose()里，要么多一对括号
plt.imshow(TtoP)

# transform6: 图像颜色高斯波动


'''--------------------------------------------------------
Q. 如何利用torchvision的datasets导入数据？
- 首先还是下载数据集，然后可以利用pytorch统一化的API来方便地导入这些数据集
- 可用数据集的API：
    MNIST
    Fashion-MNIST
    EMNIST
    COCO
        Captions
        Detection
    LSUN

    Imagenet-12
    CIFAR10
    CIFAR100
    STL10
    SVHN
    PhotoTour
-----------------------------------------------------------
'''
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5],
                                                     std=[0.5,0.5,0.5])])
# 导入CIFAR10
trainset = datasets.CIFAR10(root = '/Users/suliang/MyDatasets/CIFAR10/',
                            transform = transform,
                            train = True,
                            download = True)
testset = datasets.CIFAR10(root = '/Users/suliang/MyDatasets/CIFAR10/',
                           transform = transform,
                           train = False,
                           download = True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')
data, labels = trainset[0]

'''--------------------------------------------------------
Q. 如何导入非标数据集？
- 使用如下两个通用API：
    ImageFolder
    DatasetFolder
-----------------------------------------------------------
'''







'''--------------------------------------------------------
Q. 如何分包数据？
-----------------------------------------------------------
'''
# 分包数据
trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size = 4,
                                          shuffle = True,
                                          num_workers = 2)
testloader = torch.utils.data.DataLoader(testset, 
                                          batch_size = 4,
                                          shuffle = True,
                                          num_workers = 2)