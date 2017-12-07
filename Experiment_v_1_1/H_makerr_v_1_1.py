#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys


import numpy as np
np.set_printoptions(threshold=np.inf)

def make_h(xp,w):
    #h = (w.T) * xp
    h=w.T.dot(xp)
    return h