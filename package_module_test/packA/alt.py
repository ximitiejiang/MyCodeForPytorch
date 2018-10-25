#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 12:12:19 2018

@author: ubuntu
"""





class Test():
    def p():
        print('this is model_a class test!')


def set():
    print('model_a.py')
    
    
if __name__=='__main__':
    #from . import apple  # 为什么不能相对导入
    from ..packB import bar
    bar.set()
    #from ..packB import bar  # 为什么不能相对导入
