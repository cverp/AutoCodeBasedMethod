#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import S_maker_v_1_1 as S_m
import V_makerr_v_1_1 as V_m
import W_makerr_v_1_1 as W_m
#import X_readerr_v_1_1 as X_r
np.set_printoptions(threshold=np.inf)

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
def entre_ACBM(alpha,beta,x,w,v,s):
    s_aim = s
    w_aim = w
    v_aim = v
    s_past =np.mat(np.random.random(size=(s[:,0].size,s[0].size)))
    s_past[0,0]=s[0,0]-123.0
    w_past =np.mat( np.random.random(size=(w[:,0].size,w[0].size)))
    w_past[0, 0] = w[0, 0] - 123.0
    v_past = np.mat(np.random.random(size=(v[:,0].size,v[0].size)))
    v_past[0, 0] = v[0, 0] - 123.0
    ii=1
    wf = open('../log/main_entrance_log.txt', 'w')
    print >> wf, '%s\n\n' % ("log:")
    wf.close()
    while ((get_max_Positon(v_aim-v_past))>0.01 or ((get_max_Positon(s_aim-s_past))>0.01 and True) or (get_max_Positon(w_aim-w_past))>0.01  ):
        if ((get_max_Positon(s_aim - s_past)) < 0.0001):
            print("s is ok")
        if ((get_max_Positon(w_aim - w_past)) < 0.0001):
            print("w is ok")
        if ((get_max_Positon(v_aim - v_past)) < 0.0001):
            print("v is ok")
        v_past = v_aim
        v_aim=V_m.make_v(x,w_aim)

        w_past = w_aim
        w_aim = W_m.make_w(beta, x, v_aim, s_aim)

        s_past = s_aim
        s_aim = S_m.make_s(alpha, beta, x, w_aim)

        flag=np.linalg.norm(x-v_aim*w_aim.T*x)
        flag_max_v = get_max_Positon(v_aim-v_past)
        flag_max_s = get_max_Positon(s_aim - s_past)
        flag_max_w = get_max_Positon(w_aim - w_past)
        str_flag=str(ii)+":main_flag(x-y)\t"+str(flag)+"\tflag_v:\t"+str(flag_max_v)+"\tflag_s:\t"+str(flag_max_s)+"\tflag_w:\t"+str(flag_max_w)
        print(str_flag)
        wf = open('../log/main_entrance_log.txt', 'a')
        print >> wf, '%s\n\n' % (str_flag )
        ii=ii+1
    #print(s_aim)
    wf.close()
    wf = open('../aim/w_aim.txt', 'w')
    print >> wf, '%s\n\n' % (w_aim)
    wf.close()
    wf = open('../aim/w_past.txt', 'w')
    print >> wf, '%s\n\n' % (w_past)
    wf.close()
    wf = open('../aim/s_aim.txt', 'w')
    print >> wf, '%s\n\n' % (s_aim)
    wf.close()
    wf = open('../aim/s_past.txt', 'w')
    print >> wf, '%s\n\n' % (s_past)
    wf.close()
    wf = open('../aim/v_aim.txt', 'w')
    print >> wf, '%s\n\n' % (v_aim)
    wf.close()
    wf = open('../aim/v_past.txt', 'w')
    print >> wf, '%s\n\n' % (v_past)
    wf.close()
    return w_aim,s_aim,v_aim






if __name__ == '__main__':

    k=20
    matname="../data/orl/orl(28-23).mat"
    matlable="orl"
    #xp=X_r.read_x(matname,matlable )
    k=2
    xp = np.mat("1.0 2 1 7 8;0 2 8 9 0;4 5 9 5 6;0 2 8 11 0;17 5 12 5 6")
    SIZE_x = xp[:,0].size
    SIZE_y = xp[0].size
    s = np.mat(np.random.random(size=(SIZE_y, SIZE_y)))
    w = np.mat(np.random.random(size=(SIZE_x, k)))
    v = np.mat(np.random.random(size=(SIZE_x, k)))


    alpha =beta =1.0
    entre_ACBM(alpha, beta, xp, w, v, s)
    print("hello word")

