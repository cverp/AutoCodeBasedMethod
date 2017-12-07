#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys


import numpy as np
import L_makerr_v_1_1 as L_m
import D_maker_v_1_1 as D_m
np.set_printoptions(threshold=np.inf)


def make_w(beta,x,v,s):
    print("make w_aim")
    l = L_m.make_l(s)
    SIZE_l = l[0].size
    SIZE_x = x[:,0].size
    e=np.eye(SIZE_l,dtype=float)
    #print(e)
    e1 = np.eye(SIZE_x, dtype=float)
    p=e+((beta/2.0)*(l+l.T))
    a = x.dot(p.dot(x.T))#+0.000001*e1
    b=x.dot(x.T.dot(v))
    aa= np.linalg.pinv(a)
    #w = np.linalg.solve(a, a)
    #w = np.linalg.solve(a, b)
    #w = (np.linalg.inv(a)).dot(b)
    w = aa.dot(b)
    print("w_aim ok")
    return w

if __name__ == '__main__':
    x = np.mat("1 2 1 7 8;0 2 8 9 0;4 5 9 5 6;0 2 8 11 0;17 5 12 5 6")
    s = np.mat("1 2 1 7 8;0 2 8 9 0;4 5 9 5 6;0 2 8 9 0;4 5 9 5 6")
    v = np.mat("1 0 0 ;0 2 0 ;0 5 0 ;0 0 8 ;0 5 9 ")
    d=D_m.make_d(s)
    print(make_w( 1.0,x,v,s ))