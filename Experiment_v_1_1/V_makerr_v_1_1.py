#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

import numpy as np
np.set_printoptions(threshold=np.inf)



def make_v(xp,w):
    print("make v_aim")
    A = w.T * xp * xp.T
    a,f,b=np.linalg.svd(A,full_matrices=False)
    v=b.T.dot(a.T)
    print("v_aim ok")
    #print(v[:, 0].size)
    #print(v[0].size)
    return v


if __name__ == '__main__':
    x = np.mat("1 2 1 7 8;0 2 8 9 0;4 5 9 5 6;0 2 8 9 2")
    w = np.mat("1 0 0; 0 1 0; 1 1 0; 0 0 1")
    make_v( x, w, )