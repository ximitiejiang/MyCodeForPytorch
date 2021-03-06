#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 21:52:36 2018

@author: suliang

当前主机配置：
    - CPU: intel i7 7800, 6核12线程，主频3.5GHz
    - GPU: NVIDIA GTX1080ti, 显存11G, 2块
    - 内存: 镁光16G DDR4 2400
    - 主板: intel X229芯片组, LGA 2066 ATX, 支持双路PCI-E 16X
    - 硬盘: 256G固态硬盘
    - 电源: 额定1000W 80plus

"""









'''
Q. 如何使用pytorch自带的自动梯度autograd计算？
- 所有深度学习框架都是基于计算图，pytorch基于计算图开发了反向传播自动求导引擎
  能够根据输入/前向传播过程，来自动构建计算图，自动反向传播，自动求导
- 用户需要把tensor封装成variable：variable中包括三个东西data/grad/grad_fn
- 用户需要创建前向传播过程：设计forward()函数
- 用户需要调用根结点的backward()方法：调用y.backward()函数
- pytorch就会构建计算图，反向传播计算，对叶子节点自动求导
'''
# 只需要导入Variable类对tensor进行封装，然后对标量父节点执行backward()函数
# 就会沿着计算图的树结构求解叶子结点的梯度,。
# 但注意叶子节点梯度会一直保留，所以每个epoch最后需要对梯度清0
# Variable类跟tensor类几乎一样，API也几乎一样，所有tensor换成variable一般也能正常运行。
# 可以把Variable理解成是tensor的一个wrapper，在tensor基础上增加了grad和创建该variable的function
# 所以variable可以在backward()运行后自动更新梯度

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
Q. 如何自动求导的细节分析
* 每个tensor都有一个flag为requires_grad，用于设定该tensor是否需要自动求导，默认为False
    - tensor默认requires_grad=False; 预训练模型默认requires_grad=True
    - 可用该flag来冻结相关参数的梯度计算
* 自动求导机理：基于程序前向计算过程，创建一个计算图，以叶子节点为输入，根结点为输出，
    - 通过从从根结点反向跟踪到叶子节点。
    - 自动求导代表了这张计算图，堪称一个function函数，可通过apply()来计算结果
    - 在前向计算过程中，就会同步评估梯度来构建反向计算图
    - 每个tensor的.grad_fn属性用于进入该梯度计算
    - 每一个循环都会重新创建计算图, 但由于叶子节点梯度会一直保留，其他节点都会计算完自动清0
      所以需要手动清零：

* 函数backward(gradient=None, retain_graph=None, create_graph=False)：
    - 核心参数1: gradient = None，即默认只计算标量的梯度，如果是矢量，需要指定(gradient=Tensor)
    - 该函数只计算保留叶子结点的梯度，其他节点梯度计算完就释放不保留
    - 该函数保留了叶子结点梯度，为了下次循环的重新计算，需要手动清零叶结点梯度
* 非叶子结点计算完后梯度会被清空，获得非叶节点梯度的方法是autograd.grad或hook技术
'''
# 案例：如何设置部分参数不做自动求导，部分参数做自动求导
model = torchvision.models.resnet18(pretrained=True) # 迁移学习一个模型
for param in model.parameters():
    param.requires_grad = False   #冻结模型所有梯度计算
model.fc = nn.Linear(512, 100)    # 替代模型的全联接层，默认需要自动求导
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)  #优化器也只优化全联接层

# 案例：查看反向求导路径上哪些变量会求导，受哪些因素影响是否求导？
import torch
x = torch.randn(3, 3)  # x为叶节点，默认不进行梯度计算
y = torch.randn(3, 3)  # y为叶结点，梯度保留
z = torch.randn((3, 3), requires_grad=True)   # c为中间节点，梯度不保留

a = x + y
b = a + z.sum()    # d为根结点，梯度不保留

print(x.requires_grad, y.requires_grad, z.requires_grad, a.requires_grad, b.requires_grad)  # 初始状态，默认都要求导
print(x.is_leaf, y.is_leaf, z.is_leaf, a.is_leaf, b.is_leaf)
print(x.grad, y.grad, z.grad, a.grad, b.grad)  # backward之前，所有grad都为None
b.backward()
print(a.grad, b.grad, c.grad, d.grad)  # 为什么没有梯度计算结果：回过头去看陈云的书

# 案例：自动求导
x = torch.ones(1, requires_grad=True)
w = torch.rand(1, requires_grad=True)
y = x * w
# y依赖于w，而w.requires_grad = True
x.requires_grad, w.requires_grad, y.requires_grad


'''
Q. 如何获得自动求导过程中不保留的中间节点的梯度值？
'''
# 方法1: 采用autograd.grad函数
x = torch.ones(3, requires_grad=True)
w = torch.rand(3, requires_grad=True)
y = x * w
z = y.sum()
print(torch.autograd.grad(z, y))       # z对y的梯度，隐式调用backward()

# 方法2: 采用hook钩子获得梯度，此处暂时留着（参考实例）
grad_list = []
def hook(grad):   # 用于传入注册hook函数的形参函数，该形参函数的输入必须是grad
    grad_list.append(grad)
    print(grad_list)

x = torch.ones(3, requires_grad=True)
w = torch.rand(3, requires_grad=True)
y = x * w
h = y.register_hook(hook)  #对预求解梯度的变量注册hook:注意register_hook只接受一个函数为形参
z = y.sum()
z.backward()  # 在执行backward函数时，就会调用对y的hook
h.remove()    # 如果不需要了，需要移除hook


'''
Q. 如何理解自动求导的单次执行，二次就会报错的原理？
- 前向计算建立时(即定义x,y的求解公式)，就会同步建立计算图用于反向求导计算
- 一次前向计算过程，对应一次backward计算，backward计算完成后，计算图就释放清空
- 如果特殊情况，需要一次前向计算，多次反向求导，则设置retain_graph=true保留计算图
'''

x = torch.ones(1)
b = torch.rand(1, requires_grad = True)
w = torch.rand(1, requires_grad = True)
y = w * x # 等价于y=w.mul(x)
z = y + b # 等价于z=y.add(b)

z.backward(retain_graph=True)  #backward参数为保留计算图，所以下次backward不会报错
w.grad

z.backward()  # backward参数为空，所以计算图会释放，不再能进行反向求导
w.grad

z.backward()  # 计算图已经释放，计算报错。
w.grad


'''
Q: 如何利用backward()的求导，来计算一些求导运算？
'''
# 计算y = x^2 * exp(x)的导数
import torch
x = torch.randn(3,4, requires_grad=True)
y = x**2 * x.exp()
y.backward(torch.ones(y.size()))
print(x.grad)

# 直接计算值: 与跟backward自动求解的梯度值相同
dydx = 2*x*x.exp() + x**2*x.exp()
print(dydx)


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
Q. 如何定义每种特殊的层（如何理解torch.nn模块）？
'''
# 有各种不一样的层：都放置在torch.nn中
# nn.module是一个基类，用于定义构建所有自己的类
# nn.Sequntial()是一个容器，用于定义
import torch.nn as nn

# 全联接层
# 默认前两个数字直接写：输入特征数，输出特征数；基本没其他参数
dens = torch.nn.Linear(14*14*128,1024)

# 二维卷积层
# 参数(in_channels,out_channels,kernel_size,strid=1,padding=0,dilation=1,groups=1,bias=True)
# 默认前三个数字直接写：输入层数，输出层数，核大小；其他参数需要名称说明
conv = torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1)

# 最大池化层
# 参数(kernel_size,stride=None,padding=0,dilation=1,return_indices=False,ceil_mode=False)
# 默认前一个数字直接写：核大小；其他参数需要名称说明
m = torch.nn.MaxPool2d(stride=2, kernel_size=2)

# 激活层
m = torch.nn.ReLU()




import torch.nn as nn
# 单层模型
l1 = nn.Linear(2, 2)  # 创建2输入，2输出全联接层
l1.modules()     # 显示只有一个模型

l1.state_dict()
for key, value in l1.state_dict().items(): # 单层模型，迭代取出字典的键和值
    print(key, value)
    
for para in l1.parameters():   # 单层模型取出可迭代模型参数tensor
    print(para.data)           # 打印参数
l1.weight           # 只有一个模型，所以直接可以调用weight和bias参数
l1.bias             # 如果是多个模型的容器，则会封装成dict放在state_dict()中

# 多层模型
model = nn.Sequential(nn.Linear(2,2), nn.Linear(2,2))
print(model)     # model可以直接打印
print(model[0])  # model也可以切片打印
for i, m in enumerate(model.modules()):  # model.modules取出来后可以单独打印某层 
    print(i,'->',m)

for para in model.parameters():   # 取出模型的所有参数：可迭代，tensor
    print(para.data, para.size())  # 显示参数值和参数尺寸

model.state_dict().keys()         # 取出模型状态参数：可迭代，字典
for key, value in model.state_dict().items(): # 迭代取出字典的键和值
    print(key, value)


'''
Q. 如何组合多个层形成一个网络？
'''
# 方法1: 用sequential类来组合各个层的类
import torch
models = torch.nn.Sequential(torch.nn.Linear(32, 32),
                             torch.nn.ReLU(),
                             torch.nn.Linear(32, 64))
print(models)
print(models[0])   # 通过默认编号切片取出该层
# 方法2: 用orderdict类来传入层参数：多了对每一个层模型的自定义名字
# 这种方法优点在于：可以直接引用名称，来调用该层
# 说明：pytorch默认对层的组合采用的是list方式，所以能用切片访问方法model[0]
# 说明：而如果用orderedDict方式，则可用字典访问方式，所以能用model.l1
import torch
from collections import OrderedDict
models = torch.nn.Sequential(OrderedDict([('l1', torch.nn.Linear(32, 32)),
                                        ('ReLU', torch.nn.ReLU()),
                                        ('l2', torch.nn.Linear(32,64))]))
print(models)
print(models.l1)  # 通过名称取出该层

'''
Q. 如何继承一个已有pytorch的模型并应用？
- 使用torch.nn.module，其中module是一个数据结构，可以表示某层，也可表示一个神经网络
- 
'''
# 继承模型：从torch.nn.Module这个类库中来
import torch
class Model(torch.nn.Module):  # 定义一个类（从torch.nn.Module继承过来）
    def __init__(self):
        super(Model, self).__init__()
        
# 继承数据集：从torch.utils.data这个类库中来
from torch.utils import data
class MyDataSet(data.Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms


'''
Q. 如何选择合适的损失函数？
'''
# 似乎Pytorch需要定义loss，然后会自动关联到model的计算中去？
# loss得到的一般是标量，可以用于backward计算梯度值：loss.backward()
loss = torch.nn.MSELoss()
loss = torch.nn.L1Loss()
loss = torch.nn.CrossEntropyLoss()


'''
Q. 如何选择合适的optimizer梯度求解器？
- 模型的参数需要传递给求解器：model.parameters()
- 理解model.parameters()：他是每层w和b的汇总，每层w或b分别为一个tensor, 所以5层一共10个tensor, len(list(model.parameters()))=10  
- 用一个参数组和一个学习率传递给求解器：
        > optim.SGD(model.parameters(),lr=0.01)
- 用多个参数组代表不同层，并规划不同学习率
        > optim.SGD([{'params':model.base.parameters()},
                     {'params':model.classifier.parameters()}],
                    lr=1e-2, monentum=0.9)
        > scheduler = StepLR(optimizer=...)
'''
# SGD: torch.optim.SGD(params, ir=, momentum=0,weight_decay=0)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
# 带动量的小批量梯度下降SGD求解器
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
# 自适应矩估计Adam求解器
optimizer = torch.optim.Adam(models.parameters(), lr=LR, betas=(0.9, 0.99))
# RMSprop 指定参数alpha
optimizer = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)


'''
Q. 如何生成指数缩小型学习率？
'''
import torch.optim.lr_scheduler.StepLR

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 设置优化器和初始学习率
# 学习率规划器：lr = 0.01, if epoch < 30
#            lr = 0.001, if 30 <= epoch < 60
#            lr = 0.0001, if 60 <= epoch < 90
scheduler = StepLR(optimizer, step_size=30, gamma=0.1) 
for epoch in range(100):
    scheduler.step()
    train()
    validate()


'''
Q. 如何调用pytorch自带的数据集？
    1. 所有数据存在 from torchvision import datasets
    2. 加载：采用trainset = datasets.MNIST(...)
    3. 直接把trainset当成可迭代对象使用： data, label = trainset[n]
    4. 或者先基于trainset生成batch
'''

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
# data_train是一个可迭代对象，等效于set(data, label)
# 所以可以直接调用： data, label = data_train[i]
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
Q. 如何打开第三方的数据集？
'''
# 针对单个图片
from PIL import Image
import matplotlib.pyplot as plt
root = '/Users/suliang/MyDatasets/DogsVSCats/train/dog.8011.jpg'
data = Image.open(root)
plt.imshow(data)

# 针对大数据集：一个文件夹
from torch.utils import data
class MyDataSet(data.Dataset):
    def __init__(self, root, transforms=None):  # 拼接每个文件的地址
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms
    
    def __getitem__(self, index):  # 基于输入index，计算label, 处理img
        img_path = self.imgs[index]
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label
    
    def __len__(self):
        return len(self.imgs)
    
root = '/Users/suliang/MyDatasets/DogsVSCats/train' 
data_train = MyDataSet(root, transforms = None)
img, label = data_train[2]  # 切片相当于调用__getitem__()的调用

import matplotlib.pyplot as plt
plt.imshow(img)

# 针对大数据集：每一类一个文件夹
# 此时可用ImageFolder
data_dir = '/Users/suliang/MyDatasets/hymenoptera_data'
# 读取数据
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
# 分包数据
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}



'''
Q. 如何显示图片？
'''
# 单张图片：获取 - 还原 - 转秩 - 显示
import matplotlib.pyplot as plt
import numpy as np
data, label = trainset[102]        # tensor 3x32x32
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


'''
Q. 如何在显示图片时实现图片维度的改变：增减维度，维度顺序调整？
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image   # pytorch推荐的打开图片方法, 也是事实上的python标准图像处理库
root = '/Users/suliang/MyDatasets/DogsVSCats/train/dog.8011.jpg'
img = Image.open(root)
transform = transforms.ToTensor()
data = transform(img)

# 用img.size 查看图片本身的尺寸(像素H,W)
# 用data.size() 查看图片tensor的尺寸(像素C,H,W)

# 增加维度: squeeze是指挤的意思，unsqueeze代表放松，也就是加维度
data = data.unsqueeze(0)

# 增加维度
data = data.view(1,3,56,56)

# 调整维度顺序

# 减小维度(还原维度)
data = data.squeeze(0)

# 还原维度顺序


'''
Q. 如何处理文件夹的相对路径和绝对路径？
- os.listdir: 获得目录下所有文件的文件名
- os.path.dirname: 返回目录的目录名
- os.path.exists: 
- os.path.isdir: 
- os.path.isfile: 
- os.path.samefile: 
- os.path.split: 拆分
'''
import os

# 生成每个文件的绝对地址
root = '/Users/suliang/MyDatasets/DogsVSCats/train'
imgs = os.listdir(root)   # 输入的root必须是绝对地址
imgs = [os.path.join(root, img) for img in imgs]  # 拼接地址

# 拆分地址，获得文件名或者部分描述
imgs[0].split('/')[-1]   # 基于'/'拆分，并获得最后一个
if 'dog' in imgs[0].split('/')[-1]:
    print('this is dog!')
    

'''
Q. 如何通过函数对图形进行变换(放大缩小，旋转，裁剪，等等)？
'''



'''
Q. 如何使用pytorch自带的高级模型？
- Pytorch自带了AlexNet, VGG, ResNet, Inception
- 使用时的要求参考原文：https://github.com/pytorch/vision/blob/master/docs/source/models.rst
  使用pytorch自带的模型有一些注意事项：
      >部分模型需要区分训练模式和验证模式
      >所有预训练模型都期望输入的图形都基于相同标准：数值在[0,1]之间，顺序CxHxW
      >所有图形先做标准化，normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
'''
import torchvision.models as models

alexnet = models.alexnet(pretrained=True)           # 下载完成, 244M
# 深度加深到8层
# 真正把深度神经网络带入大众的王者归来，让深度学习完全超越传统机器学习
# 第一次采用dropout来预防过拟合
# 第一次采用GPU

vgg16 = models.vgg16(pretrained=True)               # 下载完成, 553M (网络加深到)
# 深度加深到16层的VGG16和19层的VGG19
# 统一了卷积层参数：卷积核3x3，step=1, Padding=1
# 第一次采用更小卷积核(从5x5减小到3x3)

vgg19 = models.vgg19(pretrained=True)


inception = models.inception_v3(pretrained=True)    # 下载完成, 109M (网络加深到20层，参数大小却极大减小)
# 也叫GoogleNet，深度加深到22层(V3) 
# 第一次取消全联接层，所以节省了运算减少参数(googleNet参数只有AlexNet的一半)
# 第一次采用Inception单元结构: 把数据并行送入4种卷积池化核（1x1卷积, 3x3卷积, 5x5卷积, 3x3池化）后再合并
# 优化的Inception v3则在卷积池化核之前先增加1x1卷积来做特征通道的聚合，能够有效减少所需参数的数量
# 其中1x1卷积核借用了NIN模型(network in network)用于保持空间维度，降低深度

resnet18 = models.resnet18(pretrained=True)         # 下载完成, 47M (网络进一步加深到上百层，参数大小却极小)
# 深度进一步加深, 可以有18层的resnet18, 甚至50层/101层/152层
# 引入残差模块(恒等映射)，进一步解决高层数的梯度消失问题(比普通ReLU更有效)，真正让网络达到上百甚至上千层
# 残差模块就是添加短路连接shortcut，可以让网络更加深
#
# 
resnet34 = models.resnet34(pretrained=True)             # 下载

resnet50 = models.resnet50(pretrained=True)           # 待下载

restnet101 = models.resnet101(pretrained=True)        # 待下载

densenet = models.densenet161(pretrained=True)      # 下载完成, 115M

squeezenet = models.squeezenet1_0(pretrained=True)  # 下载完成, 5M


print(alexnet)

# 导入torchsummary查看模型的输出形状，模型参数个数，模型参数大小
# 安装torchsummary: pip3 install torchsummary
from torchsummary import summary
summary(vgg16, input_size=(3,244,244))

summary(resnet18, input_size=(3,244,244))  # ? 对输入维度有什么要求

summary(alexnet, input_size=(3,64,64))     # ? 对输入维度有什么要求


'''
Q: 如何用可视化工具visdom?
    - 启动visdom: $ python -m visdom.server
    - 打开visdom: http://localhost:8097
'''
import torch as t
import visdom
vis = visdom.Visdom(env=u'test1')    # 构建一个客户端对象vis
x = t.arange(1,30,0.01)          
y = t.sin(x)

vis.line(X=x, Y=y, win='sinx', opts={'title':'y=sin(x)'}) 
#在vis里边绘制



'''
Q: 一些基本的linux, GPU命令

- flops = the number of floating-point multiplication-adds per second，
  即浮点数先乘后加的能力，
'''
nvidia-smi   # 查看显存占用(显存大，所能运行的网络也大)，
             # 查看GPU占用情况(GPU大，计算越快)


pip3 install gpustat  # 安装gpustat，他基于nvidia-smi，








