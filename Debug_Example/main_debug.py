#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 22:12:13 2018

@author: ubuntu
"""

try:
    import ipdb

except:
    import pdb as ipdb


def sum(x): # 求和函数
    r = 0
    for ii in x:
        r += ii
    return r

def mul(x): # 求乘积函数
    r = 1
    for ii in x:
        r *= ii
    return r

ipdb.set_trace()  # 该句会自动进入debug模式

x = [1,2,3,4,5]
r = sum(x)
r = mul(x)