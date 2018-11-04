#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 11:40:21 2018

@author: ubuntu
"""

#import torch
#a = torch.ones(3,4)
#b = torch.zeros(3,4)
#
#c = a.add(b)
#
#
#
#x = torch.randn((3,4), requires_grad=True)
#y = x**2*x.exp()
#
#x.grad
#y.grad
#
#x.requires_grad
#y.requires_grad
#
#y.backward(torch.ones(3,4))
#x.grad
#y.grad
#
#
#x = torch.tensor([1.0, 2.0], requires_grad = True)
#
#y = x*2
#y.register_hook(print)
#
#z = torch.mean(y)
#z.backward()
#x.grad
#y.grad

# ------------------------------------------
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyMul(nn.Module):
    def forward(self, input):
        out = input * 2
        return out
    
class MyMean(nn.Module):            # 自定义除法module
    def forward(self, input):
        out = input/4
        return out

def tensor_hook(grad):
    print('tensor hook')
    print('grad:', grad)
    return grad

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.f1 = nn.Linear(4, 1, bias=True)    
        self.f2 = MyMean()
        self.weight_init()
    def forward(self, input):
        self.input = input
        output = self.f1(input)       # 先进行运算1，后进行运算2
        output = self.f2(output)      
        return output
    def weight_init(self):
        self.f1.weight.data.fill_(8.0)    # 这里设置Linear的权重为8
        self.f1.bias.data.fill_(2.0)      # 这里设置Linear的bias为2
    def my_hook(self, module, grad_input, grad_output):
        print('doing my_hook')
        print('original grad:', grad_input)
        print('original outgrad:', grad_output)
        # grad_input = grad_input[0]*self.input   # 这里把hook函数内对grad_input的操作进行了注释，
        # grad_input = tuple([grad_input])        # 返回的grad_input必须是tuple，所以我们进行了tuple包装。
        # print('now grad:', grad_input)        
        return grad_input
if __name__ == '__main__':
    input = torch.tensor([1, 2, 3, 4], dtype=torch.float32, requires_grad=True).to(device)
    net = MyNet()
    net.to(device)
    net.register_backward_hook(net.my_hook)   # 这两个hook函数一定要result = net(input)执行前执行，因为hook函数实在forward的时候进行绑定的
    input.register_hook(tensor_hook)
    result = net(input)
    print('result =', result)
    result.backward()
    print('input.grad:', input.grad)
    for param in net.parameters():
        print('{}:grad->{}'.format(param, param.grad))