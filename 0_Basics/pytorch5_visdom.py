#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:30:34 2018

@author: ubuntu
"""

'''--------------------------------------------------------
Q. 如何应用visdom?

需要预安装visdom: 本机已经装过了
    - 把所有图片放在一个界面上显示，便于查看(比之前用plt.plot绘制的更集成)
    - visdom帮助文件： https://github.com/facebookresearch/visdom
                      https://www.pytorchtutorial.com/pytorch-visdom/    
    - 安装后需要启动visdom服务：python -m visdom.server, 作为web server默认绑定8097端口
    - 开启服务后，绘图，然后就可以在http://localhost:8097上查看
    - visdom同时支持tensor和numpy array, 但不支持python的int，float这类标量
    - 绘图函数包括：line, image, text, hisgram, scatter, bar, pie
    - 基础绘图：
      1. 创建环境: vis = visdom.Visdom(env='aaa')
      2. 计算x,y: x=torch.arange(1,30,0.1), y=x
      3. 绘图: vis.line(X=x,Y=y,win='yx',opts={'title':'xx',
                                               'xlabel':'xx',
                                               'ylabel':'xx'})
      4. 绘图不覆盖：首先win需要相同名称，其次有两种实现方法
          增加update='append'参数
          增加update='new'参数
-----------------------------------------------------------
'''
import visdom
# 创建环境
vis = visdom.Visdom(env=u'123')  # 建立环境env=test1, u代表utf-8字符编码，避免在非utf-8系统下发生乱码
# 创建窗口，绘制一根线条
x = torch.arange(1,100,0.01)
y = torch.sin(x)
vis.line(X=x, Y=y, win='sinx', opts={'title':'y=sin(x)'})
# 创建窗口，绘制n根线条
for ii in range(0,10):
    x=torch.tensor([ii])
    y=x
    vis.line(X=x,Y=y,win='aaa', update='append' if ii>0 else None)
# 创建窗口，绘制叠加线条
x=torch.arange(0,9,0.1)
y=(x**2)/9
vis.line(X=x,Y=y,win='aaa', name='this is a new trace', update='new')
# 创建窗口，显示图片
vis.image(torch.randn(256,256), win='bbb')  # HxW, 或者CxHxW即可
vis.image(torch.randn(3,512,512),win='ccc')
# 创建窗口，显示文字
vis.text('hello world', win='ddd', opts={'title':'Visdom title show'})

