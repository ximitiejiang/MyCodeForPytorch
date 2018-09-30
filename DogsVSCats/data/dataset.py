#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 16:54:34 2018

@author: suliang

数据集导入

参考：深度学习框架Pytorch入门与实践
"""

'''
导入数据
'''
from torch.utils import data
import os
from PIL import Image
import numpy as np
from torchvision import transforms as T

   
class DogCat(data.Dataset):
    
    def __init__(self, root, transforms=None, train = True, test = False): 
        '''
        定义dataset的初始化内容
        负责取出所有图片文件名，定义训练集/验证集/测试集，定义数据变换方式
        '''
        self.train = train
        self.test = test
        
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        
        if self.test: # 如果是测试集，则按照文件名中第二个数字字段排序
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:    # 如果是训练集，则按倒数第2个字段数字排序
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        
        
        
        imgs_num = len(imgs)        
        if self.test:  # 如果是测试集，则全部取出
            self.imgs = imgs
        elif train:   # 如果是训练集，则取出0.7的数据
            self.imgs = imgs[:int(0.7*imgs_num)]
        else:  # 如果是验证集，则取出0.3的数据
            self.imgs = imgs[int(0.7*imgs_num)]
        
        if transforms is None:
            
            # 为了代码形式简洁，先定义比较长的一个normalize： 为什么mean, std是这个值？
            normalize = T.Normalize(mean = [0.485, 0.456,0.406],
                                    std = [0.229, 0.224,0.225])
            
            # 如果是测试集或者验证集（非训练集）: 把数据标准化处理，为什么多一个centerCrop?
            if self.test or not self.train: 
                self.transforms = T.Compose([T.resize(224),
                                             T.CenterCrop(224),
                                             T.ToTensor(),
                                             normalize])
            # 如果是训练集：尽可能增加数据量，所以增加随机切割，随机翻转  
            else:
                self.transforms = T.Compose([T.resize(224),
                                             T.RandomSizeCrop(224),
                                             T.RandomHorizontalFlip(),
                                             T.ToTensor(),
                                             normalize])

    
    def __getitem__(self, index):  # 基于输入index，计算label, 处理img
        '''
        定义dataset的切片返回内容
        负责返回一张图片和对应标签
        如果是测试集，没有标签，则返回图片和标题ID
        '''
        img_path = self.imgs[index]
        if self.test:  # 测试数据，返回文件名的标题ID
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:  # 如果是训练或者验证数据，返回标签
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
            
        data = Image.open(img_path)
        data = self.transforms(data)
        
        return data, label
        
    def __len__(self):
        '''
        定义dataset.len的返回内容
        '''
        return len(self.imgs)


# 配置基本信息    
train_data_root = '/Users/suliang/MyDatasets/DogsVSCats/train' 
test_data_root = '/Users/suliang/MyDatasets/DogsVSCats/train' 
load_model_path = 'checkpoints/model.pth'

batch_size = 128
use_gpu = True
num_workers = 4
print_freq = 20  # 每n个batch打印一次

debug_file
result_file

max_epoch = 10
lr = 0.1
lr_decay = 0.95          # lr = lr*lr_decay
weight_decay = 1e-4      # 损失函数？

data_train = DogCat(root, transforms = transform)

'''
单张预览
'''
img, label = data_train[4]    # 切片相当于调用__getitem__()的调用
pic = img*0.5 + 0.5
import matplotlib.pyplot as plt
pic = np.transpose(img,(1,2,0))
plt.imshow(pic)

'''
单批次预览
'''
x_example, y_example = next(iter(dataloader['train']))
print(x_example.shape, y_example.shape)


model = models.vgg16(pretrained = True)
print(model)
