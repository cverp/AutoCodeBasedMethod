#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def get_max_Positon(aim):#获取最大
    a = aim
    raw, column = a.shape  # get the matrix of a raw and column
    max_positon = np.argmax(a)  # get the index of max in the a
    m, n = divmod(max_positon, column)
    pax=abs(a[m,n])
    min_positon = np.argmin(a)  # get the index of max in the a
    k, j = divmod(min_positon, column)
    pin = abs(a[k, j])
    p=0.0
    if(pax>pin):
        p= pax
    else:
        p=pin
    return p

if __name__ == '__main__':
    m=np.mat("2 3; 4 5")
    n = np.mat("2 3; 1 15")
    ss=m-n
    print(np.sum(m, axis=1))
    print(get_max_Positon(ss))
    print(get_max_Positon(m-n))
    print(n ,m ,ss)
