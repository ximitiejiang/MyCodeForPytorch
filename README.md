# MyCodeForPytorch

# 卷积操作是什么？
    - 通过卷积核与输入进行扫描，每个扫描窗口计算对应位置相乘后求和。
    - 卷积核心功能是提取图像特征
    - 参考动画dl_3_12.gif:
    - 对灰度单层图片卷积：采用1个卷积核 + 1个偏置，输出1层数据
    - 对彩色3层图片卷积：采用3个卷积核 + 1个偏置，输出1层数据
    - 对n层中间数据卷积：采用n个卷积核 + 1个偏置，输出1层数据（参考LeNet5的S2 to C3）
    
# 卷积的优点？
    - 稀疏连接
    - 权值共享
    
# 卷积(CNN)与全连接(BP神经网络)的差别？
    - 全连接的每个输入跟下一层的每个神经元都关联
    - 卷积的每个输入只跟下一层部分神经元关联

# 卷积神经网络的发展
    - LeNet5是第一代卷积神经网络
    - AlexNet是第二代，也是真正流行起来的王者卷积神经网络
    - 接下来CNN分成四个发展方向
    - 方向1: 

1998 - LeNet-5，由LeCun提出，是第一个真正意义上的卷积神经网络
    - 首次提出了卷积(conv)，下采样(subsampling)，非线性激活三大特性
        
2012 - AlexNet，由Alex Krizhevsky，Hinton提出，
    - 提出ReLU函数替代sigmoid函数
    - 提出dropout技术避免过拟合
    - 提出max pooling技术
        
2014 - VGGNet，由牛津大学VGG视觉几何小组提出
    - 提出使用小卷积3x3(而不是之前的大卷积5x5或者更大)
        
2014 - GoogleNet，由谷歌提出
    - 
2014 - DeepFace, 由Taigman提出
    - DeepID, 由汤晓鸥提出
       
2014 - R-CNN, 由加州伯克利教授 Jitendra Malik提出
    - ？

2015 - Fast R-CNN, 由RCNN第一作者 Ross Girshick提出
    - 
2015 - ResNet，由何凯明提出
    - 解决了训练极深网络时梯度消失的问题
2016 - Faster R-CNN, 由微软的孙剑、任少卿、何凯明、Ross Girshick提出
    - 
2017 - Mask R-CNN
    - 由Facebook AI 的何凯明、Girshick提出
    - 把Faster R-CNN拓展到像素级的图像分割

