#!/usr/bin/env python3
# -*- coding: utf-8 -*-   

from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体  
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题 

import sys

import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np 

from sklearn.cluster import KMeans
from sklearn.externals import joblib
np.set_printoptions(threshold=np.inf)  




mat_file='Results_ACBM_Clustering.mat'
mat_name='label'
data_name="orl"
data = sio.loadmat(mat_file)  # 读取mat文件
p=data.keys()
Results_ACBM_Clustering = {}
Results_ACBM_Clustering[data_name] = data[data_name]
print(((Results_ACBM_Clustering[data_name])[data_name + 'w']))   # 查看mat文件中的所有变量


wf = open('result.txt', 'a')
print >> wf, '%s\n\n' % (p )
'''
matrix1 = data['gnd'] 
matrix2 = data[mat_name]
matrix5 = data['AC']
SIZE_MAT1=matrix1.size
matrix3 = data['res']
#feature = [[float(x) for x in row[0:]] for row in matrix2]
#调用kmeans类
SIZE_MAT=matrix1.size
downnum=float(SIZE_MAT)
accuracy=0.0
for i in range(0,SIZE_MAT):
	if(matrix1[0][i]==matrix3[0][i]):
		accuracy=accuracy+1.0
accuracy=accuracy/downnum


print(matrix1)
print('Execute done.')
print(matrix2)
print('Execute done.')
print(matrix3)
print('Execute done.')
print(accuracy*downnum)
print(accuracy)

#plt.plot([1,2,3,4], [1,4,9,16], 'ro')
#plt.axis([0, 6, 0, 20])
#plt.show()
'''
#s = np.mat(np.array([[(x) for x in range(0, 8)] for l in range(0, 3)]))
s = data['orl']
print(s)
print('Execute done.')
