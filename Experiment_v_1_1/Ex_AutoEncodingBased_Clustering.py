#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
import scipy.io as scio
mpl.rcParams['font.sans-serif'] = ['SimHei']
np.set_printoptions(threshold=np.inf)


def read_x(mat_file,mat_name):
    data = scio.loadmat(mat_file)  # 读取mat文件
    x = np.mat(data[mat_name])
    print("Get data---done")
    return x

def make_d(s):
    s_1 = np.sum(s, axis=1)
    s_2 = np.sum(s, axis=0)
    s1=np.array((s_1.T+s_2))
    d = (np.diag(s1[0]))/2.0
    return d

def make_l(s):
    SIZE_s = s[0].size
    s_ = np.mat(np.array([[0.0 for xs in range(0, SIZE_s)] for ls in range(0, SIZE_s)]))
    for i in range(0, SIZE_s):
        for j in range(0, SIZE_s):
            s_[i, j] = s[i, j]*s[i, j]
    d = make_d(s_)
    l=d-s_
    return l

def make_h(xp,w):
    h=w.T.dot(xp)
    return h

def make_v(xp,w):
    print("make v_aim")
    A = w.T*xp*xp.T
    a,f,b=np.linalg.svd(A,full_matrices=False)
    v=b.T.dot(a.T)
    print("v_aim ok")
    return v

def make_w(beta,x,v,s):
    print("make w_aim")
    l = make_l(s)
    SIZE_l = l[0].size
    e=np.eye(SIZE_l,dtype=float)
    p=e+((beta/2.0)*(l+l.T))
    a = x.dot(p.dot(x.T))
    b=x.dot(x.T.dot(v))
    aa= np.linalg.pinv(a)
    #w = np.linalg.solve(a, a)
    #w = np.linalg.solve(a, b)
    #w = (np.linalg.inv(a)).dot(b)
    w = aa.dot(b)
    print("w_aim ok")
    return w


def make_s(alpha,beta,xp,w):
    print("make s_aim")
    SIZE_x = xp[0].size
    h=make_h(xp,w)
    s = np.mat(np.array([[0.0 for xs in range(0, SIZE_x)] for ls in range(0, SIZE_x)]))
    a = np.mat(np.array([[0.0 for xa in range(0, SIZE_x)] for la in range(0, SIZE_x)]))
    a1 = np.mat(np.array([[0.0 for xa1 in range(0, SIZE_x)] for la1 in range(0, SIZE_x)]))
    a2 = np.mat(np.array([[0.0 for xa2 in range(0, SIZE_x)] for la2 in range(0, SIZE_x)]))
    temp=xp.T.dot(xp)
    temp1 = h.T.dot(h)
    for i in range(0,SIZE_x):
        for j in range(0, SIZE_x):
            if(i!=j):
               a1[i, j] = temp[i, i] + temp[j, j] - 2 * temp[i, j]
               a2[i, j] = temp1[i, i] + temp1[j, j] - 2 * temp1[i, j]
               if((alpha*a1[i,j]+beta*a2[i,j])!=0):
                    a[i,j]=1.0/(alpha*a1[i,j]+beta*a2[i,j])
               else:
                   a[i,j]=0.0
            else:
                a[i,j]=0.0
    a_sum = np.sum(a, axis=1)
    for j in range(0,SIZE_x):
        for i in range(0, SIZE_x):
            if(i!=j and a_sum[i,0]!=0):
                s[i,j]=((a[i,j])/(a_sum[i,0]))
            else:
                s[i,j]=0.0
    #print(s)
    print("s_aim ok")
    return s

def paint_chart(y,y_name):
    y_len=len(y)
    names=[xc for xc in range(0, y_len)]
    x = range(len(names))
    plt.plot(x, y,  mec='r', mfc='w')
    plt.legend()  # 让图例生效
    #plt.xticks(x, names, rotation=45)
    #plt.margins(0)
    #plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"time(s)")  # X轴标签
    plt.ylabel(y_name)  # Y轴标签
    plt.title(y_name)  # 标题
    plt.show()


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
    flag_list=[]
    flag_aim_list = []
    flag_w_list = []
    flag_s_list = []
    flag_v_list = []
    wf = open('../log/main_entrance_log.txt', 'w')
    #print >> wf, '%s\n\n' % ("log:")
    wf.write("%s\n\n" % "log:")
    wf.close()
    aim_value=0.0001
    #aim_value = 0.01
    while ((get_max_Positon(v_aim-v_past))>aim_value or ((get_max_Positon(s_aim-s_past))>aim_value ) or (get_max_Positon(w_aim-w_past))>aim_value or (ii<=100 and 1)  ):
        s_past = s_aim
        s_aim = make_s(alpha, beta, x, w_aim)
        w_past = w_aim
        w_aim = make_w(beta, x, v_aim, s_aim)
        v_past = v_aim
        v_aim = make_v(x, w_aim)
        l_aim=make_l(s_aim)

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
        str_flag=str(ii)+":main_flag(x-y)\t"+str(flag)+"\tflag_v:\t"+str(flag_max_v)+"\tflag_s:\t"+str(flag_max_s)+"\tflag_w:\t"+str(flag_max_w)
        print(str_flag)
        wf = open('../log/main_entrance_log.txt', 'a')
        #print >> wf, '%s\n\n' % (str_flag )
        wf.write('%s\n\n' % (str_flag))
        ii=ii+1

    return w_aim,s_aim,v_aim,flag_list,flag_aim_list,flag_w_list,flag_s_list,flag_v_list


def ACBM(k,matname,dataset_name):
    xp=read_x(matname,dataset_name )
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
    paint_chart(flag_list,"norm(x-v_aim*w_aim.T*x)")
    paint_chart(flag_aim_list, "||...||+alpha||...||+beta||...||")
    paint_chart(flag_w_list, "w-w_past")
    paint_chart(flag_v_list, "v-v_past")
    paint_chart(flag_s_list, "s-s_past")

if __name__ == '__main__':
    k=20
    matname1 = "../data/orl/orl(28-23).mat"
    matname2 = "../data/ar10/ar10.mat"
    matname3 = "../data/mnist/mnist(28-28).mat"
    dataset_name1 = "orl"
    dataset_name2 = "ar"
    dataset_name3 = "mnist"
    ACBM(k, matname2, dataset_name2)
    print("hello word")

