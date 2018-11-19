#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 16:11:48 2018

@author: ubuntu
"""

'''--------------------------------------------------------
Q. 如何使用模型的__init__()函数和forward()函数？
-----------------------------------------------------------
'''
# 模型初始化就会自动调用__init__()函数
import torch
class AAA(torch.nn.Module):
    def __init__(self, x):
        super(AAA,self).__init__()
        print('This is class init! print: {}'.format(x))
    
    def forward(self,x):
        print('This is class foward! print: {}'.format(x))

aaa = AAA(99)  # 在类中传入的参数： 创建类的对象，就会自动调用__init__()函数

outputs = aaa(258)  # 在对象中传入的参数： 调用对象默认方法，就会自动调用forward()函数
                    # forward()是对象默认方法，aaa(258) = aaa.forward(258)


'''--------------------------------------------------------
Q. 如何使用继承模型已有的backward()函数？
-----------------------------------------------------------
'''





'''--------------------------------------------------------
Q. 如何定义优化器optimizer？
-----------------------------------------------------------
'''
# SGD优化器：实际上为mini-batch随机梯度下降求解器，并带有动量参数
# SGD也可以带weight_decay
optimizer = optim.SGD(model.parameters(), 
                      lr=0.001, 
                      momentum=0.9)
# Adam优化器：自适应矩估计
# Adam计算流程：引入belta1=0.9,belta2-0.999,
# 先算梯度，然后算mt(一阶矩估计，累积梯度指数衰减均值), vt(二阶矩估计，累积平方梯度平均值)
# 再基于mt,vt来做偏差校正得到mt_hat, vt_hat, 最后得到梯度更新

optimizer = optim.Adam(model.parameters(),
                       lr = opt.lr,
                       weight_decay = opt.weight_decay)


'''--------------------------------------------------------
Q. 如何定义损失函数criteria？
1.损失函数是什么：就是loss函数，用来表示预测值与标签值之间的偏差的函数，对损失函数进行梯度计算来得到预测函数的参数。
  为了能够求出参数，损失函数的设计要考虑设计成便于求梯度，有极大值的凸函数。
2.常见机器学习损失函数：
    - LiR/线性回归：欧式距离，等于min欧式距离，最后欧式距离求极值转化为梯度计算求参数
    - LoR逻辑回归：最大似然估计，等于最大似然函数即max联合概率，取log连乘变连加，最后最大似然函数求极值转化为梯度计算求参数
    - Softmax回归：交叉熵，等于min交叉熵，最后交叉熵求极值转化为梯度计算求参数
    - SVM分类：
    - 决策树分类：损失函数是极大似然，但实现时是用最优特征选择，即信息增益或基尼指数，即递归方式通过最大信息增益或者最小基尼指数选择最优特征。
3.深度学习损失函数：
    - 均方误差损失函数torch.nn.MSELoss
    - 负log似然损失函数torch.nn.NLLLoss
    - 交叉熵损失函数torch.nn.CrossEntropyLoss
    
4.求得的loss到底什么样子？什么含义？
    - loss得出的输出是一个tensor(float)，即带标量的tensor，也就是一个单值.
    - loss代表了1个batch的每张图片的平均损失, loss*batch_size就是一个batch的总loss.
-----------------------------------------------------------
'''
# 1. 定义损失函数
# MSE损失函数：即均方误差损失函数
criterion = nn.MSELoss()
# NLLLoss损失函数：即负log似然损失，用于多分类问题，需要配合最后添加一层log softmax layer
criterion = nn.NLLLoss()
# 交叉熵损失函数：整合nn.LogSoftmax()和nn.NLLLoss()在一个class，所以不需要配合添加log softmax layer
criterion = nn.CrossEntropyLoss()

# 2. 计算损失（基于outputs）
outputs = model(inputs)  # 计算输出
loss = criterion(outputs, labels)  # 一个batch的平均损失，得到的是一个单数值的tensor
running_loss += loss.item() * inputs.size(0)  # 累加每个batch的总损失
epoch_loss = running_loss / len(trainloader.dataset)  # 一个epoch的平均损失

# 另一种计算损失的方法(也是基于outputs,但用loss_meter.add(loss.data[0]))
outputs = model(inputs)  # 计算输出
loss = criterion(outputs, labels)  # 一个batch的平均损失，得到的是一个单数值的tensor
loss_meter.add(loss.data[0])  # batch loss累加得到epoch loss


'''--------------------------------------------------------
Q. 探讨下交叉熵损失函数到底做了什么事？
-----------------------------------------------------------
'''
from torch import nn
outputs = torch.randn(3, 5, requires_grad=True)
labels = torch.empty(3, dtype=torch.long).random_(5)

#ct1 = nn.LogSoftmax()
#ct2 = nn.NLLLoss()
#loss = ct1(outputs, labels)

# crossentropyloss = logsoftmax + nllloss
ct3 = nn.CrossEntropyLoss()
loss = ct3(outputs, labels)
loss.backward()






'''--------------------------------------------------------
Q. 如何计算精度accuracy?

计算流程： 注意每步中数据类型的变化， tensor(矩阵) - tensor(标量) - 标量
1. 计算outputs
2.1 计算loss:  outputs -> loss -> running_loss -> epoch_loss
2.2 计算acc :  outputs ---------> running_acc --> epoch_acc
-----------------------------------------------------------
'''
# 计算精度（基于outputs）
outputs = model(inputs)                            # 1. 计算预测输出：4xnclass
_, preds = torch.max(outputs, 1)                   # 2. 计算预测概率：筛选出其中max value
running_corrects += torch.sum(preds == labels.data) # 3. 累加一个epoch正确数
epoch_acc = running_corrects.double() / len(trainloader.dataset)  # 计算精度

# 另一种表达精度的方法（也是基于outputs, 但用confusion matrix表示）
outputs = model(inputs)                            
confusion_matrix.add(outputs.data, labels.data)



'''--------------------------------------------------------
Q. 如何训练一个简单的迁移模型？
对于如果使用的pytorch做迁移模型，需要把所有输入图片size转换成224x224，除了inception为299,
需要把图片基先转化到90,1)之间(这一步可以在ToTensor的transforms做)
然后基于mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]进行normalization, 
详细参考：https://pytorch.org/docs/master/torchvision/models.html
-----------------------------------------------------------
'''




