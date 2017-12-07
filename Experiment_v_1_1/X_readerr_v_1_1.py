#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys


import numpy as np
import scipy.io #as sio

np.set_printoptions(threshold=np.inf)


def read_x(mat_file,mat_name):
    data = scipy.io.loadmat(mat_file)  # 读取mat文件
    x = np.mat(data[mat_name])
    #print(x)
    return x


if __name__ == '__main__':
    read_x("../data/coil(32-32).mat","label")
    '''
        A = np.mat("1 -2 1;0 2 -8;-4 5 9")
        print "A\n", A

        b = np.mat("0 8 -9;0 8 -9;0 8 -9    ")
        print "b\n", b
        x = np.linalg.solve(A, b)
        print "Solution", x
        pp=np.linalg.norm(x, 2)  # 谱范数,为x里最大值开平方
        print(pp)    
        x =  np.mat(np.array([[1 for x in range(0, n)] for x in range(0, n)]))
        m = np.linalg.solve(x, x)
        #m = np.linalg.inv(x)#求逆
        print(m)
        print(x)
        '''
    print("loaded")

