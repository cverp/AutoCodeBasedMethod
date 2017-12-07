#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import numpy as np
import D_maker_v_1_1 as D_m
np.set_printoptions(threshold=np.inf)


def make_l(s):
    SIZE_s = s[0].size
    s_=np.mat(np.random.random(size=(s[:,0].size,s[0].size)))
    for i in range(0, SIZE_s):
        for j in range(0, SIZE_s):
            s_[i, j] = s[i, j]*s[i, j]
    d = D_m.make_d(s_)
    l=d-s_
    #print(l)
    return l

if __name__ == '__main__':
    s = np.mat("1 2 1 7 8;0 2 8 9 0;4 5 9 5 6;0 2 8 9 0;4 5 9 5 6")

    print(make_l( s ))
