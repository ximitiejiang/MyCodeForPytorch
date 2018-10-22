#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:11:18 2018

@author: ubuntu
"""

'''--------------------------------------------------------
Q. pytorch报错：Expected object of type torch.FloatTensor 
but found type torch.cuda.FloatTensor for argument #2 'weight'

1. 代码报错位置：
> /home/ubuntu/suliang_git/MyCodeForPytorch/0_Basics/test_module.py(116)<module>()
    114         optimizer.zero_grad()
    115         # 计算每个batch的输出/损失/预测
--> 116         outputs = model(inputs)
3   117         loss = criteria(outputs, labels)
    118         _, preds = torch.max(outputs, 1)

2. 问题解析：输入到model去的变量为cuda变量，但系统期望一个cpu变量
但实际情况是，我已设置device = cuda：0，且已设置inputs.to(device).
查了下inputs的device属性inputs.device, 发现返回的是cpu形式，但我确实是把data和label
都设置成cuda形式过了。会过头去查看代码发现：
        inputs.to(device)
        labels.to(device)
这种写法不会改变变量的属性，因为to命令不是inplace方式，需要赋值操作，改为：
        inputs = inputs.to(device)
        labels = labels.to(device)
-----------------------------------------------------------
'''
# 错误写法
inputs.to(device)
labels.to(device)
# 正确写法
inputs = inputs.to(device)
labels = labels.to(device)

