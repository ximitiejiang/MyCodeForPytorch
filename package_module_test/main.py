#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:12:03 2018

@author: ubuntu
"""

'''

2. 自动生成的__pycache__文件夹什么用？是否影响模块的重新加载？
  用于存放python脚本编译后的代码，如果脚本没有变化，下次就直接执行该文件夹内的编译版本，节省时间
  如果不想要生成__pycache__, 可以在执行时用python -B aaa.py
  如果想永远不产生，可以设置环境变量 PYTHONDONTWRITEBYTECODE=1

'''

# ----------方式1：----------
import packA     # 直接导入包不行，因为模块没有导入，所以无法运行
packA.alt.set()     
# ----------方式2：----------
import packA.alt
packA.alt.set()
# ----------方式3：----------
from packA import alt   # 导入模块可行，但需要引用时带着包名
packA.alt.set()    # 引用模块中的函数
# ----------方式4：----------
from packA.alt import Test
test = Test()

