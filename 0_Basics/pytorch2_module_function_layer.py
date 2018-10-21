#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:11:09 2018

@author: ubuntu
"""


'''--------------------------------------------------------
Q. 如何定义Module? 
-----------------------------------------------------------
'''
import torch
import torch.nn as nn

# 创建模型方法1
class AAA(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        
    def forward(self, x):
        x = conv1(x)
        x = nn.ReLU(x)
        x = conv2(x)
        x = nn.ReLU(x)
        
net = AAA()

# 创建模型方法2：默认是用list的方式存放各层，调用各层就是对list切片
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
for i in range(len(model)):
    print(model[i])

# 创建模型方法3: 优点是对模型取名存成字典，调用各层就是对dict切片
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
          ]))
model.conv2


'''--------------------------------------------------------
Q. module定义好以后有哪些可用的属性？
这些属性在后续的train中都会使用到
-----------------------------------------------------------
'''
import torch.nn as nn    
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU())
model.modules()          # 输出模型里边所有子模型，包括sequential()模型和层模型
model.to(device)         # 把模型送到运行设备
    model = model.to(device)
model.parameters()       # 返回所有参数tensor
    for param in model.parameters():
        param.requires_grad = False
model.named_parameters() # 返回所有参数tensor的[名称, tensor]
    for name, tensor in model.named_parameters():
        print(name)
model.state_dict()       # 输出模型状态字典：包括参数和缓存的{名字和数值}，{key:value}
    
model.train()            # 设置模型在训练模式
model.eval()             # 设置模型在测试模式
model.zero_grad()        # 设置模型所有参数梯度归零    
    

'''--------------------------------------------------------
Q. 如何定义各个层? 
-----------------------------------------------------------
'''    
import torch
import torch.nn as nn
# 卷积层： [输入层数，输出层数，默认bias=True]
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
input = torch.randn(20, 16, 50, 100)
output = m(input)

m = nn.MaxPool2d(3, stride=2)
m = nn.MaxPool2d((3, 2), stride=(2, 1))
input = torch.randn(20, 16, 50, 32)
output = m(input)

m = nn.AvgPool2d(3, stride=2)
m = nn.AvgPool2d((3, 2), stride=(2, 1))
input = torch.randn(20, 16, 50, 32)
output = m(input)

# 非线性激活
m = nn.Softmax()
input = torch.randn(2, 3)
output = m(input)

m = nn.ReLU()
input = torch.randn(2)
output = m(input)

m = nn.BatchNorm2d(100)
m = nn.BatchNorm2d(100, affine=False)
input = torch.randn(20, 100, 35, 45)
output = m(input)

# 直接有RNN模块和LSTM模块在torch.nn中

# 线性模块: [输入层数，输出层数，默认bias=True]
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())

# dropout层：[关闭概率p]
m = nn.Dropout2d(p=0.2)
input = torch.randn(20, 16, 32, 32)
output = m(input)


'''--------------------------------------------------------
Q. 如何设计一个迁移模型? 
-----------------------------------------------------------
'''  
import torch
import torch.nn as nn
from torchvision import models

# 定义resnet18模型
model = models.resnet18(pretrained=True)
print(model)
in_fc = model.fc.in_features # 获得自定义fc层的输入特征层数
model.fc = nn.Linear(in_fc, 2) # 自定义fc层, 输出为2个分类

# 定义vgg16模型
model = models.vgg16(pretrained=True)
print(model)
in_fc = model.classifier[6].in_features
model.classifier[6] = nn.Linear(in_fc, 2)  # 自定义fc层，输出为2分类

    
'''--------------------------------------------------------
Q. 如何设计一个resnet18? 
-----------------------------------------------------------
'''  
import torch
import torch.nn as nn

class Resnet(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        
        
    def forward(x):
        



