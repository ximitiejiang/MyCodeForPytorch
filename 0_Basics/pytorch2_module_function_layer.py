#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:11:09 2018

@author: ubuntu
"""

'''
Q. 如何合理选择funtional和module模块？
'''
# 1. 注意区分同样是relu在functional里边和在torch.nn里边大小写是不同的
import torch.functional as F
import torch.nn as nn
F.relu()
nn.ReLU()
# 2. 在torch.nn里边定义的层，在类型上属于nn.module，能够被自动提取可学习参数，同时可以使用module的功能，比如model.train(), model.eval()
#    因此，对于有参数的层(conv, bn, linear)都应使用nn.module，放在init函数中自动提取可学习参数。
#    对于受train/eval影响的层(dropout)也应使用nn.module.
#    而其他无可学习参数的曾，可以使用functional，放在forward函数中，两者性能没有区别。
nn.Dropout()
nn.BatchNorm2d()
nn.Linear()
nn.Conv2d()
nn.ReLU()



'''--------------------------------------------------------
Q. 如何定义Module? 
-----------------------------------------------------------
'''
import torch
import torch.nn as nn

# 创建模型方法1
class AAA(torch.nn.Module):
    def __init__(self):
        super(AAA, self).__init__()
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
- model可以通过切片调用
- parameters怎么调用：model.parameters()循环调用，model.named_parameters()循环调用
-----------------------------------------------------------
'''
import torch.nn as nn    
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU(),
          nn.Linear(64,10))
# model的切片和调用：
model[1]  # 调用层：用Sequential声明的model
model.fc  # 调用层：用OrderedDict声明的model
model[2].bias  # 调用参数
# 输出模型里边所有子模型，包括sequential()模型和层模型
model.modules()   
# 把模型送到运行设备       
model.to(device)         
    model = model.to(device)
# 模型参数方法: 返回一个generator对象，可迭代取出tensor
model.parameters()       
    for param in model.parameters():
        param.requires_grad = False
        print(param)
    for param in model[4].parameters():
        print(param)
# 模型的名称和参数方法：返回一个generator对象，可迭代取出[名称, tensor]
model.named_parameters() 
    for name, layer_para in model.named_parameters():
        print(name, layer_para.data.std(), layer_para.grad.data.std())
        print(name)
    named = []
    for name, _ in model.named_parameters():
        named.append(name)
    print(named)
        
# 输出模型状态字典：包括参数和缓存的{名字和数值}，{key:value}
model.state_dict()       
# 设置模型在训练模式,测试模式    
model.train()            
model.eval()   
# 设置模型所有参数梯度归零           
model.zero_grad()           

   
'''--------------------------------------------------------
Q. 如何理解激活函数？ 
-----------------------------------------------------------
''' 
# 激活函数是把线性特征非线性化？
# 最常用激活函数是ReLU，因为他的导数是1, 不会导致梯度消失或者梯度爆炸
# 但ReLU会把所有负的值都截断为0,所以在回归问题上一般不用。





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
Q. 如何设计一个vgg16? 
以下为自己改写的pytorch vgg16
-----------------------------------------------------------
'''  
from models.Basicmodule import BasicModule  # 相对导入
import torch.nn as nn


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3  # 初始通道数为3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG16(BasicModule):

    def __init__(self, num_classes=2):
        
        super().__init__()
        
        self.model_name = 'vgg16'  # 创建一个名称属性，用于保存model

        cfg ={11 : [64,     'M', 128,      'M', 256, 256,      'M', 512, 512,                'M', 512, 512,           'M'],
              13 : [64, 64, 'M', 128, 128, 'M', 256, 256,      'M', 512, 512,                'M', 512, 512,           'M'],
              16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,           'M', 512, 512, 512,      'M'],
              19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}
        
        # 如果要用VGG19,则修改为cfg[19],
        # 默认增加BN层，如果不要BN层，则改为make_layers(cfg[16], batch_norm=False)
        self.features = make_layers(cfg[16])
        
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(True),
                                        nn.Dropout(),
                                        nn.Linear(4096, num_classes),
                                        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

        
'''--------------------------------------------------------
Q. 研究一个最简CNN，比如alexnet后，对nn.module, 计算输出，更新梯度，更新参数的总结？
-----------------------------------------------------------
'''  
model = nn.Sequential(nn.conv2d(1,6,5,1,1),
                      nn.conv2d(6,10,5,1,0),
                      nn.Linear(400,120),
                      nn.Linear(120,84),
                      nn.Linear(84,10))
criterion = torch.nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

optimizer.zero_grad()
outputs = model(inputs)            
loss = criterion(outputs, labels)     # 这一步有可能很复杂的扩展
loss.backward()    # 更新梯度                     
optimizer.step()   # 更新参数

'''运行上面这个小网络的过程总结：
1. model, sequential都是基于nn.module基类, nn.module基类的特点：
   重写了__setattr__()：
   重写了__getattr__()：
   重写了__call__(): 会调用forward()，所以每一个nn.module子类都需要重写forward()函数用于实例的调用

2. 添加子模型的过程：先调用__setattr__(),函数内判断是module, 则调用基类module.add_module()
   把每个子模型都添加到_modules变量中存储起来。
   添加过程：如果是sequential()模型，则for idx, module in enumerate(model) ...
   如果是sequential(OrderedDict())模型，则for key, module in args[0].items ...

3. 添加模型参数(w,b)的过程：先调用__setattr__()，函数内判断是Parameters,则调用register_parameters()
   把每个参数添加到_parameters变量中存储起来
   
4. 所以子类继承nn.module基类的好处是：模型可自动添加到_modules统一做forward计算
   参数可以自动添加到_parameters统一做参数更新(optimizer.step())

5. 其他loss类/criterion类继承nn.module目的我的理解是沿袭更新forward()，实例化时__call__()
   就是调用forward()的nn.module()

6. 模型训练的5步法：
    optimizer.zero_grad(): 用于梯度清零，在module的zero.grad()函数中，对_parameters中每个.grad都清零
    outputs = model(inputs): 用forward()计算单次正向输出
    loss = criterion(outputs, labels)：用损失函数公式计算输出损失
    loss.backward()：更新梯度
    optimizer.step()：更新参数

7. 进一步拆解5步，实现相对小的实例！！！加hook...
   深刻管控每步的输入输出，考虑hook的使用？考虑单步输出？

'''
