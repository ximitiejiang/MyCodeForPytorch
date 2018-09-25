#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 09:56:33 2018

@author: suliang

创建一个单隐藏层的多层感知机

"""

# 导入

###################################
# 最简单的单隐层多层感知机：手动计算梯度
###################################
import torch
def simple_MLP():
    # 定义网络结构
    batch_n = 100
    hidden_layer = 100
    input_data = 1000
    output_data = 10
    
    # 初始化数据
    x = torch.randn(batch_n, input_data)
    y = torch.randn(batch_n, output_data)
    
    # 初始化权重
    w1 = torch.randn(input_data, hidden_layer)
    w2 = torch.randn(hidden_layer, output_data)
    
    # 训练参数
    epoch_n = 50
    learning_rate = 1e-6
    
    # 开始训练
    for epoch in range(epoch_n):
        h1 = x.mm(w1)
        h1 = h1.clamp(min = 0)   # 类似与ReLu激活函数(截断小于0的数据，大于0的数据 y=x)
        y_pred = h1.mm(w2)   # 计算预测值
        
        loss = (y_pred - y).pow(2).sum()   # 损失
        print('Epoch: {}, Loss:{:.4f}'.format(epoch, loss))
        
        grad_y_pred = 2*(y_pred - y)  # 计算输出
        grad_w2 = h1.t().mm(grad_y_pred)
        
        grad_h = grad_y_pred.clone()
        grad_h = grad_h.mm(w2.t())
        grad_h.clamp_(min = 0)
        grad_w1 = x.t().mm(grad_h)
        
        w1 -= learning_rate*grad_w1
        w2 -= learning_rate*grad_w2
        
###################################
# 优化的单隐层多层感知机：引入自动梯度计算，引入Variable封装tensor，(免去自己写反向传播和梯度计算)
###################################
import torch
from torch.autograd import Variable        
def MLP_autograd():
    # 定义网络结构
    batch_n = 100
    hidden_layer = 100
    input_data = 1000
    output_data = 10
    
    # 初始化数据: 用variable类来对已定义的tensor数据进行封装
    x = Variable(torch.randn(batch_n, input_data), requires_grad = False)
    y = Variable(torch.randn(batch_n, output_data), requires_grad = False)
    
    # 初始化权重
    w1 = Variable(torch.randn(input_data, hidden_layer), requires_grad = True)
    w2 = Variable(torch.randn(hidden_layer, output_data), requires_grad = True)

    # 训练参数
    epoch_n = 20
    learning_rate = 1e-6
    
    # 开始训练
    for epoch in range(epoch_n):
        h1 = x.mm(w1)
        h1 = h1.clamp(min = 0)   # 类似与ReLu激活函数(截断小于0的数据，大于0的数据 y=x)
        y_pred = h1.mm(w2)   # 计算预测值
        
        loss = (y_pred - y).pow(2).sum()   # 损失计算
        print('Epoch: {}, Loss:{:.4f}'.format(epoch, loss))
        
        loss.backward()   # backward()函数可用于Variable封装的数据
        
        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data

        
        w1.grad.data.zero_()
        w2.grad.data.zero_()   



###################################
# 优化的单隐层多层感知机：引入模型的继承，免去自己写反向传播和梯度计算
###################################
        
class Model(torch.nn.Module):  # 定义一个类（从torch.nn.Module继承过来）
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, input, w1,w2):
        x = torch.mm(input, w1)
        x = torch.clamp(x, min = 0)
        x = torch.mm(x, w2)
        return x
    
    def backward(self):
        pass
    
def MLP_inherit_class():
    # 定义网络结构
    batch_n = 100
    hidden_layer = 100
    input_data = 1000
    output_data = 10
    
    # 创建一个模型对象
    model = Module()
    
    # 初始化数据: 用variable类来对已定义的tensor数据进行封装
    x = Variable(torch.randn(batch_n, input_data), requires_grad = False)
    y = Variable(torch.randn(batch_n, output_data), requires_grad = False)
    
    # 初始化权重
    w1 = Variable(torch.randn(input_data, hidden_layer), requires_grad = True)
    w2 = Variable(torch.randn(hidden_layer, output_data), requires_grad = True)

    # 训练参数
    epoch_n = 30
    learning_rate = 1e-6
    
    # 开始训练
    for epoch in range(epoch_n):
        y_pred = model(x,w1,w2)   # 用继承的模型对象进行前向计算
        
        loss = (y_pred - y).pow(2).sum()   # 损失
        print('Epoch: {}, Loss:{:.4f}'.format(epoch, loss))
        
        loss.backward()
        
        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data
        
        w1.grad.data.zero_()
        w2.grad.data.zero_()
    

###################################
# 优化的单隐层多层感知机：借用torch自带的梯度求解器
###################################
def MLP_Adam():
    import torch
    from torch.autograd import Variable
    
    # 定义网络结构
    batch_n = 100
    hidden_layer = 100
    input_data = 1000
    output_data = 10
    
    # 初始化数据: 用variable类来对已定义的tensor数据进行封装
    x = Variable(torch.randn(batch_n, input_data), requires_grad = False)
    y = Variable(torch.randn(batch_n, output_data), requires_grad = False)
    
    models = torch.nn.Sequential(torch.nn.Linear(input_data, hidden_layer),
                                 torch.nn.ReLU(),
                                 torch.nn.Linear(hidden_layer, output_data))
    epoch_n = 100
    learning_rate = 1e-4
    loss_fn = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)
    
    # 开始训练
    for epoch in range(epoch_n):
        y_pred = models(x)   # 
        
        loss = loss_fn(y_pred, y)
        print('Epoch: {}, Loss:{:.4f}'.format(epoch, loss.data[0]))
        optimizer.zero_grad()  # 对模型参数梯度归零
        
        loss.backward()
        
        optimizer.step()


        
if __name__ == '__main__':
    
    test_id = 3
    
    if test_id == 1:
        simple_MLP()
    
    elif test_id == 2:
        MLP_autograd()
    
    elif test_id == 3:
        MLP_Adam()
    
    else:
        print('Wrong test_id!')

