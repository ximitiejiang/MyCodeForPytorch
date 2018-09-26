#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 21:52:36 2018

@author: suliang

pytorch的基础知识整理：Variable-->autograd-->

1. 介绍下几个典型的卷积神经网络：
    * 卷积层：用于对输入数据进行特征提取，主要依靠里边的卷积核，卷积核就是一个指定窗口大小的扫描器。
    * 卷积核：
    * 池化层：用于对输入数据的核心特征提取，可以实现对原始数据的压缩，还能大量减少参与模型计算的参数
    * 平均池化层
    * 最大池化层
    * 

2. 新建数据类型
    * torch.FloatTensor(2,3)    浮点数
    * torch.IntTensor([2,3,4])   整数
    * torch.Tensor([1,2,3])     随意指定生成内容
    * torch.from_numpy([1,2,3])
    
    * torch.rand(2,3)   随机浮点数, 均匀分布，(0, 1)之间
    * torch.randn(2,3)   随机浮点数, 标准正态分布，0均值，1方差
    * torch.range(1,20,1)   顺序数
    * torch.zeros(2,3)
    
3. 数据计算
    * torch.abs(a)   取绝对值
    * torch.add(a,b)   相加(对应位置相加)
    * torch.clamp(a, -0.1, 0.1)   裁剪数据，超出裁剪边界的数据重写为裁剪边界
    * torch.div(a,b)   相除(对应位置相除)
    * torch.pow(a,2)    求幂(按位求幂)
    * torch.mul(a,b)    相乘，按位置相乘
    * torch.mm(a,b)    相乘，矩阵的点积
      a.mm(b)    也代表a与b的点积，这两种方法都可以
    * torch.mv(a,vec)    相乘，矩阵与向量的点积

4. 梯度自动计算
    * Variable    用于对tensor数据进行封装，从而可以使用系统的自动梯度计算
    * backward()   对于Variable()变量的一个方法，针对Loss使用该方法后能够产生
    * torch.autograd    自动梯度计算模块

5. torch.nn模块库：用来创建神经网络，包含了很多基本的类
    (1) 神经网络骨架    
    * torch.nn.Sequential()   用来串联各层和各激活函数，是神经网络的骨骼
    
    (2) 线性变换
    * torch.nn.Linear()    线性变换类，用于两层之间的线性变换
    
    (3) 激活函数
    * torch.nn.ReLU()    激活函数
    
    (4) 损失函数
    * torch.nn.MSELoss()    损失函数
    * torch.nn.L1Loss()    L1正则损失函数（平均绝对误差函数）
    * torch.nn.CrossEntropyLoss()    交叉熵损失函数
    
    (5) 基础模型？？？
    * torch.nn.module
    
6. torch.optim模块库：用来定义求解器optimzer，包括自动优化参数，选择求解算法
    * torch.optim.Adam()   Adam自适应梯度求解器
    * torch.optim.SGD()
    
7. torchvision.transforms模块库：用来定义数据变换器transform，来进行数据变换，数据增强
    * transforms.Compose()    用来组合各种变换方法
    * transforms.ToTensor()    用来把PIL图片数据转换成tensor变量，便于pytorch处理
    * transforms.ToPILImage()   用来把tensor数据转换回PIL图片，便于显示
    * transforms.Normalize()    用来进行数据标准化
    * transforms.Resize()
    * transforms.Scale()
    * transforms.CenterCrop()
    * transforms.

"""

#--------------Open issue---------------------
'''
Q. 如何定义基本的tensor?
'''
# 有7种基本的CPU tensor：torch.FloatTensor/torch.DoubleTensor/torch.ByteTensor/torch.CharTensor/torch.ShortTensor/torch.IntTensor/torch.LongTensor
# 最常用的2种：torch.FloatTensor, torch.IntTensor
# 默认的是FloatTensor可简写为torch.Tensor()
a = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
print(a)

b = torch.IntTensor(2, 4).zero_()
print(b)


'''
Q. 如何对tensor进行切片?
'''
import torch
x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]]) #类似于对list的多维切片,似乎也支持numpy的高级切片写法
print(x[1][2])

x[0][1] = 8   # 可以直接赋值
print(x)

print(x[:,1])  # 跟numpy一样的高级切片方式
print(x[:,1].size())


'''
Q. 如何对tensor内的元素进行计算？
'''
# 计算清零
a = torch.FloatTensor([[1,-2,3],[-4,5,-6]])
a.zero_()  # 用0填充tensor, 只有一种带后缀的方式，修改原tensor
print(a)

# 计算size()
print(a.size())  # 获得tensor的size形状

# 计算绝对值
# 需要注意带后缀下划线的函数是在原tensor上直接修改，
# 而不带后缀下划线的函数是在新的tensor操作
b = torch.FloatTensor([[1,-2,3],[-4,5,-6]])
b.abs()   # 不带后缀，不修改原tensor
print(b)
b.abs_()  # 带后缀，修改原tensor
print(b)

# 计算加法
a = torch.FloatTensor([1,2,3])
b = torch.FloatTensor([3,2,1])
c = a + b
print(c)

# 计算点积
a = torch.FloatTensor([[1,-2,3],[-4,5,-6]])
b = torch.FloatTensor([[1,2],[2,1],[0,1]])
c = a.mm(b)
print(c)

# 如果tensor不支持的操作，可先转换成numpy运算，再转换回tensor
a = torch.FloatTensor([[1,-2,3],[-4,5,-6]])
an = a.numpy()  # 把tensor转化为numpy
at = torch.from_numpy(an)  # 把numpy转化为tensor

an[0,0] = 10   # tensor与array共享内存，所以任何一个变化会引起另一个变化
print(an)
print(at)

# 矩阵的转秩
b = a.t()  # tensor转秩
print(a)
print(b)

# 计算幂次  ???
a = torch.FloatTensor([[1,-2,3],[-4,5,-6]])
b = a.pow()

# 计算求和/求平均/求最大
a = torch.FloatTensor([[1,-2,3],[-4,5,-6]])
b = a.sum()  # 求和
c = a.mean() # 求平均
d = a.max()  # 求最大
print(d)



'''
Q. 如何使用pytorch自带的自动梯度计算？
'''
# 只需要导入Variable类对tensor进行封装，然后对标量父节点执行backward()函数
# 就会沿着计算图的树结构求解叶子结点的梯度,。
# 但注意叶子节点梯度会一直保留，所以每个epoch最后需要对梯度清0
# Variable类跟tensor类几乎一样，API也几乎一样，所有tensor换成variable一般也能正常运行。
# 可以把Variable理解成是tensor的一个wrapper，在tensor基础上增加了grad和创建该variable的function
# 所以variable可以在backward()运行后自动更新梯度
# 但注意的是autograd计算梯度只能针对标量tensor，也就是backward()函数只能给标量tensor用.
# loss都是标量，所以大多数情况都是loss.backward()
# 反向传播后，会更新leaf variable即叶子节点的梯度，而不会更新父节点梯度
# 反向传播的梯度会累加，所以
import torch
from torch.autograd import Variable
w1 = Variable(torch.randn(2, 3), requires_grad = True)
print(w1)
print(w1.data)
print(w1.grad)  # 此时还没有反向传播计算，所以梯度为none

loss = torch.sum(w1)
print(loss)   
loss.backward()   # 用于backward()的对象loss必须是标量，否则报错
print(loss.grad)  # loss不是叶子节点，不会更新grad
print(w1.grad)
w1.grad.zero_()   # 对梯度清零,避免梯度不必要的累加
print(w1.grad)

# 如果要求向量自动求梯度，需要传入梯度计算参数给backward()函数
a = Variable(torch.rand(1,3),requires_grad=True)
b = a*a
print(b)
b.backward([1,1,1])


'''
Q. 如何测试Variable类的基本属性?（综合理解variable, grad, backward）
'''
# 如果要使用自动求导，就必须把tensor封装成Variable
# x, y封装成Variable之后的对象，有如下属性可以用
# x.data  为封装的tensor值
# x.grad  初始化时grad=0, 运行了y.backward()方法后，该grad值就会更新
# 调用backward()逻辑：如果调用y.backward()方法，就会对y的计算图中requires_grad=True的所有变量x求导dy/dx
# 把y的计算图理解为一棵树，y为父节点，下面有很多叶子节点，叶子结点的梯度会在反向传播中更新。

import torch
from torch.autograd import Variable
w1 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)
print('before grad:',w1.grad)
print(w1.data)

y = torch.mean(w1)  # 变换函数
y.backward()  # 基于y的计算过程，反向计算路径上每个层上过程变量的梯度，相应更新每个过程变量的x.grad
print('after grad:',w1.grad)
print(w1.data)


'''
Q. 如何进行两个tensor的点积操作？
'''
import torch
a = torch.Tensor([[1,2,3],[2,0,1]])
b = torch.Tensor([[1,2],[1,0],[1,1]])

# 以下两种写法都行，但第一种写法更方便，类似pandas的做法，跟在变量后边直接写更简洁
c = a.mm(b)    
d = torch.mm(a,b)

print(c)
print(d)


'''
Q. 如何定义每种特殊的层？
    * 线性层：
    * 卷积层
    * 池化层
    * 
'''
# 有各种不一样的层：都放置在torch.nn
import torch
torch.nn.Linear()

# 二维卷积层
m = torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)
# 最大池化层
m = torch.nn.MaxPool2d(stride=2, kernel_size=2)
# 激活层
m = torch.nn.ReLU()
# 全联接层
m = torch.nn.Linear(14*14*128,1024)


'''
Q. 如何组合多个层形成一个网络？
'''
# 方法1: 用sequential类来组合各个层的类
import torch
models = torch.nn.Sequential(torch.nn.Linear(input_data, hidden_layer),
                             torch.nn.ReLU(),
                             torch.nn.Linear(hidden_layer, output_data))
# 方法2: 用orderdict类来传入层参数
import torch
from collections import OrderedDict
models = torch.nn.Sequential(OrderedDict([('Line1', torch.nn.Linear(input_data, hidden_layer)),
                                        ('ReLU', torch.nn.ReLU()),
                                        ('Line2', torch.nn.Linear(hidden_layer, output_data))]))


'''
Q. 如何继承一个已有pytorch的模型并应用
'''
# 继承从torch.nn.Module这个类库中来
import torch
class Model(torch.nn.Module):  # 定义一个类（从torch.nn.Module继承过来）
    def __init__(self):
        super(Model, self).__init__()


'''
Q. 如何选择合适的损失函数？
'''
# 似乎Pytorch需要定义loss，然后会自动关联到model的计算中去？
# loss得到的一般是标量，可以用于backward计算梯度值：loss.backward()
loss = torch.nn.MSELoss()
loss = torch.nn.L1Loss()
loss = torch.nn.CrossEntropyLoss()


'''
Q. 如何选择合适的梯度求解器？
'''
# 需要定义一个优化器

# 带动量的小批量梯度下降SGD求解器
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
# 自适应矩估计Adam求解器
optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)


'''
Q. 如何做一个完整训练？
'''
# 初始化数据和节点数
#   * 要用Variable封装数据
#   * 
# 定义model
#   * 使用Sequential()类可以很方便定义各个层
# 定义训练准备
#   * epoch: 循环次数，每次循环都会生成一个模型并得到一个损失数据
#   * learning rate: 使用默认值
#   * loss: 可以借用系统中自带的几种损失函数，loss = torch.nn.MSELoss(y_pred, y)
#   * optimzer: 可以借用系统中自带的几种梯度求解器，optimzer = torch.optim.Adam()
# 开始训练
#   * 前向计算：
#   * 损失计算
#   * 梯度求解
#   * 反向计算


'''
Q. 如何调用pytorch自带的数据集？
'''
# pytorch在torchvision.datasets自带了MNIST，COCO（用于图像标注和目标检测），
# CIFAR10 and CIFAR100，Imagenet-12，ImageFolder，LSUN Classification，STL10
# 以上Datasets都是 torch.utils.data.Dataset的子类，
# 所以，他们也可以通过torch.utils.data.DataLoader使用多线程（python的多进程）
import torch
import torchvision
from torchvision import datasets

# 先加载进内存：采用自带datasets得到MNIST的数据对象
data_train = datasets.MNIST(root = '/Users/suliang/MyDatasets/MNIST/',
                            transform = transform,
                            train = True,
                            download = False)
# 对单个数据的调用
image,label = data_train[10]

# 如果希望分batch，则可借助DataLoader对数据进行分包和随机发送
data_loader_train = torch.utils.data.DataLoader(dataset = data_train,
                                                batch_size = 64,
                                                shuffle = True)
# 对分包的调用
images, labels = next(iter(data_loader_train))


'''
Q. 如何理解pytorch自带的分包工具DataLoader?
'''
# dataloader生成的是一个可迭代对象，调用dataloader数据的方法有很多种
# 1. 一次获得一个batch包
images, labels = next(iter(testloader))  
# 2. 循环获得每个batch包
for data in testloader:
    images, labels = data
# 3. 循环获得每个batch包并获得序号    
for i, data in enumerate(trainloader, 0):
    inputs, labels = data


'''
Q. 如何显示图片？
'''
# 单张图片：获取 - 还原 - 转秩 - 显示
import matplotlib.pyplot as plt
import numpy as np
data, label = trainset[102]   # tensor
data = data*0.5 + 0.5 
data = np.transpose(data, (1,2,0))  # 转秩，从CxHxW变为HxWxC
plt.imshow(data)

# 多张图片：获取 - 还原 - 拼接 - 转秩 - 显示
import matplotlib.pyplot as plt
import numpy as np
data, label = next(iter(trainloader))   # size = 4x3x32x32
data = data*0.5 + 0.5
data = torchvision.utils.make_grid(data)  # size = 3x36x138
data = np.transpose(data, (1,2,0))
plt.imshow(data)




