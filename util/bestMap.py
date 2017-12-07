# -*- coding: utf-8 -*-

from pylab import *
import numpy as np

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

np.set_printoptions(threshold=np.inf)

# KM算法使用的数据部分
# labe个数
n = 0
# 数组元素最大个数
N = 1000
# 负无穷
inf = float('inf')
# KM算法进行协调的数组
match = [0 for x in range(0, N)]
lx = [0 for x in range(0, N)]
ly = [0 for x in range(0, N)]
sx = [0 for x in range(0, N)]
sy = [0 for x in range(0, N)]
weight = [[0 for x in range(0, N)] for x in range(0, N)]


# last_match_list = [0 for x in range(0, 1000)]
# KM算法使用的部分结束

# 获取KM所需的数据部分

# 获取KM所需的部分结束

# KM算法使用的函数部分
def dfs(x):
    global match
    global lx
    global ly
    global sx
    global sy
    global weight
    global n
    sx[x] = True
    for i in range(0, n):
        if not sy[i] and lx[x] + ly[i] == weight[x][i]:
            sy[i] = True
            if match[i] == -1 or dfs(match[i]):
                match[i] = x
                return True
    return False


# 自己写的memset，便于全面修改代码
def memset(listt, aim, len):
    for i in range(0, len):
        listt[i] = aim
    return listt


def fax(x):
    global match
    global inf
    global lx
    global ly
    global sx
    global sy
    global weight
    global n
    if not x:
        for i in range(0, n):
            for j in range(0, n):
                weight[i][j] = -weight[i][j]
    match = memset(match, -1, N)

    for i in range(0, n):
        ly[i] = 0
        lx[i] = -inf
        for j in range(0, n):
            if weight[i][j] > lx[i]:
                lx[i] = weight[i][j]
    for i in range(0, n):
        while 1 == 1:
            sx = memset(sx, 0, N)
            sy = memset(sy, 0, N)
            if (dfs(i)):
                break
            mic = inf
            for j in range(0, n):
                if sx[j]:
                    for k in range(0, n):
                        if not sy[k] and lx[j] + ly[k] - weight[j][k] < mic:
                            mic = lx[j] + ly[k] - weight[j][k];
            if mic == 0:
                return -1
            for j in range(0, n):
                if sx[j]:
                    lx[j] = lx[j] - mic
                if sy[j]:
                    ly[j] = ly[j] + mic
    sum = 0
    for i in range(0, n):
        if match[i] >= 0:
            sum = sum + weight[match[i]][i]
    if not x:
        sum = -sum
    match_list = [0 for x in range(0, n)]
    for i in range(n):
        match_list[i] = match[i]

    return match_list


# 调用入口
def use_KM(aim_n, aim_list):
    global n
    global weight
    start = 1
    n = aim_n
    weight = aim_list
    h = fax(start)
    return h


# KM算法使用的部分结束


# 获取KM所需的二分图邻接矩阵函数部分
# 对某一label建立对应向量
def result_to_vector(code, my_matrix, SIZE_MAT):
    aim_vector = [0 for x in range(0, SIZE_MAT)]
    for i in range(SIZE_MAT):
        if my_matrix[i] == code:
            aim_vector[i] = 1
    return aim_vector


# 向量直接求点积作为graph的权重
def my_dot_product(p1, p2, SIZE_MAT):  # p1\2 are vectors
    result = 0
    for i in range(SIZE_MAT):
        result = result + (p1[i] * p2[i])
    return result


# 根据各向量及点积构造带权二分图的邻接矩阵
def all_route(aim_vec_all, al_ri_re, SIZE_MAT):
    all_result_list = [[0 for x in range(0, n)] for x in range(0, n)]
    for i in range(0, n):
        for j in range(0, n):
            all_result_list[i][j] = my_dot_product(aim_vec_all[i], al_ri_re[j], SIZE_MAT)
    return all_result_list


# 获取KM所需的二分图邻接矩阵部分结束



def predict2trueV2_labels(label_true_array, label_predict_array):
    # lable类型个数，由两者中多的那个决定
    global n
    n = 0
    n_back = 0
    SIZE_MAT = label_true_array.size

    for i in range(0, SIZE_MAT):
        if n < label_true_array[i]:
            n = label_true_array[i]
        if n_back < label_predict_array[i]:
            n_back = label_predict_array[i]

    if n < n_back:
        n = n_back
    n = n + 1
    # 邻接矩阵表示二分图的点与边
    graph_list1 = [[0 for x in range(0, n)] for x in range(0, n)]
    # 存储最终结果的数组
    label_result = np.arange(SIZE_MAT)
    # 将各元素0~n-1表述为SIZE_MAT维的向量，便于求出邻接阵
    all_right_result = [[0 for x in range(0, SIZE_MAT)] for x in range(0, n)]
    all_aim_result = [[0 for x in range(0, SIZE_MAT)] for x in range(0, n)]
    for i in range(0, n):
        all_right_result[i] = result_to_vector(i, label_true_array, SIZE_MAT)
        all_aim_result[i] = result_to_vector(i, label_predict_array, SIZE_MAT)
    graph_list1 = all_route(all_right_result, all_aim_result, SIZE_MAT)
    # 调用KM算法求匹配
    match_list = [0 for x in range(0, n)]
    match_list = use_KM(n, graph_list1)
    # 将匹配结果(匹配关系）转换为需要的结果（bestmap）
    for j in range(0, SIZE_MAT):
        p = label_predict_array[j]
        label_result[j] = match_list[p]

    return label_result


# 示例
if __name__ == '__main__':
    y = np.array([0, 0, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 1, 3, 3, 3])
    predy = np.array([0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    print(predict2trueV2_labels(y, predy))
    y = np.array([0, 1, 2])
    predy = np.array([0, 1, 1])
    print(predict2trueV2_labels(y, predy))
    y = np.array([0, 1, 1, 1, 1])
    predy = np.array([0, 2, 1, 2, 2])
    print(predict2trueV2_labels(y, predy))

