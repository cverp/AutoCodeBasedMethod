#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import S_maker_v_1_1 as S_m
import V_makerr_v_1_1 as V_m
import W_makerr_v_1_1 as W_m
import L_makerr_v_1_1 as L_m
import X_readerr_v_1_1 as X_r
import paint_chart_py as painter
import scipy.io as scio
import util.data as data_reader
np.set_printoptions(threshold=np.inf)

def get_max_Positon(aim):#获取最大
    a = aim.copy()
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
    flag_list=[]
    flag_aim_list = []
    flag_w_list = []
    flag_s_list = []
    flag_v_list = []
    wf = open('../log/main_entrance_log.txt', 'w')
    print >> wf, '%s\n\n' % ("log:")
    wf.close()
    aim_value=0.0001
    #aim_value = 0.01
    while ((get_max_Positon(v_aim-v_past))>aim_value or ((get_max_Positon(s_aim-s_past))>aim_value ) or (get_max_Positon(w_aim-w_past))>aim_value or (ii<=100 and 1)  ):

        s_past = s_aim
        s_aim = S_m.make_s(alpha, beta, x, w_aim)
        l_aim = L_m.make_l(s_aim)
        v_past = v_aim
        v_aim = V_m.make_v(x, w_aim)
        w_past = w_aim
        w_aim = W_m.make_w(beta, x, v_aim, s_aim)



        flag=np.linalg.norm(x-v_aim*w_aim.T*x)
        flag_aim =flag*flag+alpha*(np.trace(x.dot(l_aim.dot(x.T))))+beta*(np.trace(w_aim.T.dot(x.dot(l_aim.dot(x.T.dot(w_aim))))))
        flag_max_v = get_max_Positon(v_aim-v_past)
        flag_max_s = get_max_Positon(s_aim - s_past)
        flag_max_w = get_max_Positon(w_aim - w_past)
        flag_list .append(flag)
        flag_aim_list.append(flag_aim)
        flag_w_list .append(flag_max_w)
        flag_s_list .append(flag_max_s)
        flag_v_list .append(flag_max_v)
        str_flag=str(ii)+":main_flag(x-y)\t"+str(flag_aim)+"\tflag_v:\t"+str(flag_max_v)+"\tflag_s:\t"+str(flag_max_s)+"\tflag_w:\t"+str(flag_max_w)
        print(str_flag)
        wf = open('../log/main_entrance_log.txt', 'a')
        print >> wf, '%s\n\n' % (str_flag )
        ii=ii+1
    del flag_list[0]
    del flag_aim_list[0]

    return w_aim,s_aim,v_aim,flag_list,flag_aim_list,flag_w_list,flag_s_list,flag_v_list


def ACBM(k,xp,dataset_name):
    #xp=X_r.read_x(matname,dataset_name )
    SIZE_x = xp[:, 0].size
    SIZE_y = xp[0].size
    s = np.mat(np.random.random(size=(SIZE_y, SIZE_y)))
    w = np.mat(np.random.random(size=(SIZE_x, k)))
    v = np.mat(np.random.random(size=(SIZE_x, k)))
    alpha = beta = 1.0
    w_a, s_a, v_a ,flag_list,flag_aim_list,flag_w_list,flag_s_list,flag_v_list= entre_ACBM(alpha, beta, xp, w, v, s)
    Results_ACBM_Clustering = {}
    Results_ACBM_Clustering[dataset_name] = {dataset_name + 'w': w_a,
                                             dataset_name + 'v': v_a,
                                             dataset_name + 's': s_a
                                             }
    file_path = '../aim/Results_ACBM_Clustering.mat'
    scio.savemat(file_path, Results_ACBM_Clustering)
    Results_ACBM_Clustering_log = {}
    Results_ACBM_Clustering_log[dataset_name] = { 'x_log': flag_list,
                                                  'aim_log': flag_aim_list,
                                                  'w_log': flag_w_list,
                                                  'v_log': flag_v_list,
                                                  's_log': flag_s_list
                                                }
    file_path = '../aim/Results_ACBM_Clustering_log.mat'
    scio.savemat(file_path, Results_ACBM_Clustering_log)
    painter.paint_chart(flag_list,"norm(x-v_aim*w_aim.T*x)")
    painter.paint_chart1(flag_aim_list, "||...||+alpha||...||+beta||...||")
    painter.paint_chart(flag_w_list, "w-w_past")
    painter.paint_chart(flag_v_list, "v-v_past")
    painter.paint_chart(flag_s_list, "s-s_past")

if __name__ == '__main__':
    k=10
    matname1 = "../data/orl/orl(28-23).mat"
    matname2 = "../data/ar10/ar10.mat"
    matname3 = "../data/mnist/mnist(28-28).mat"
    dataset_name1 = "orl"
    dataset_name2 = "ar"
    dataset_name3 = "mnist"
    dataset_name4="alpha_digit"
    z,y=data_reader.loader(dataset='mnist')
    x=np.mat(z)
    ACBM(k, x, dataset_name3)

    print("hello word")

