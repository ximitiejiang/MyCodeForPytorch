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
          增加update='append'参数, 是在前次绘图的最后一个点为起点连接到新点，
              所以如果想要覆盖但选择append返回会多出一条从前次最后一点到该次起点的多余线条。
          增加update='new'参数，是完全覆盖前次。
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



'''--------------------------------------------------------
Q. 为什么我希望在visdom绘制accuracy的曲线却无法显示?
- 模式不对：vis.line命令支持append比较好
- 输入数据格式不对，必须是nparray/tensor,且ndim=1 or 2
-----------------------------------------------------------
'''
# 以下为初始代码
import visdom
vis = visdom.Visdom(env='main')
epoch = 1
accuracy.append(100. * (confusion_meter.value()[0][0] + confusion_meter.value()[1][1]) / (confusion_meter.value().sum()))
vis.line(X=np.arange(epoch+1), Y=accuracy, win='Acc', update='new')
# 检查发现： 报错‘win does not exist’
# 改窗口更新方式从new为append就可以显示了
vis.line(X=np.arange(epoch+1), Y=accuracy, win='Acc', update='append')
# 但append模式的问题是总会多出一条从上一轮最后一个点到本轮地一个点的多余连线
# 所以考虑不再重新绘图，而是像logger一样用append模式绘制每个epoch的点，改为如下
vis.line(X=epoch, Y=accuracy, win='cur',opts={'title':'Accuracy'}, update='append')
# 但报错说X, Y的输入ndim不对，应该是1-d或者2-d的，查了下epoch/accuracy都是标量float
# 由于visdom支持ndarray和tensor，但不支持pathon的标量，所以改为如下： 正常了。
vis.line(X=np.array(epoch).reshape(1), Y=np.array(accuracy).reshape(1), win='cur',opts={'title':'Accuracy'}, update='append')

