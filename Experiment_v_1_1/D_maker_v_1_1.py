#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np

np.set_printoptions(threshold=np.inf)


def make_d(s):
    SIZE_s = s[0].size
    s_1 = np.sum(s, axis=1)
    s_2 = np.sum(s, axis=0)
    s1=np.array((s_1.T+s_2))
    d = (np.diag(s1[0]))/2.0
    #print(d)
    return d

if __name__ == '__main__':
    #x = np.mat("1 2 1 7 8;0 2 8 9 0;4 5 9 5 6;0 2 8 9 0;4 5 9 5 6")
    x = np.mat("1 2 1 ;0 2 8 ;4 5 9")
    print(make_d( x ))

