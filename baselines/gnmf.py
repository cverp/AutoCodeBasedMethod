# -*- coding: utf-8 -*-

import numpy as np
from baselines import kmeans
from sklearn import preprocessing


# data: samples * features by default
def gnmf(data, n_components=None, nNN=5, alpha=100, max_iter=200):
    """
    :param data:
    :param n_components:
    :param nNN:
    :param alpha:
    :param max_iter:
    :return:
    """
    Dmatrix = data.T
    (M, N) = Dmatrix.shape

    # graph construction
    # nNN = 5: 1, else 0
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            W[i, j] = np.linalg.norm(Dmatrix[:, i] - Dmatrix[:, j])

    for i in range(N):
        minValues_nNN = sorted(W[i, :])[:(nNN + 1)]
        # print 'minValues_nNN', minValues_nNN, minValues_nNN[nNN]
        W[i, :][W[i, :] > minValues_nNN[nNN]] = -1

    W[W > 0] = 1
    W[W <= 0] = 0

    W = (W + W.T)  # W = 1/2 * (W + W.T)
    W[W > 0] = 1

    # L = D - W, D=diag(W*1)
    D = np.diag(np.sum(W, axis=1))

    # init Umatrix and Vmatrix
    Umatrix = np.random.uniform(0, 1, [M, n_components])
    Vmatrix = np.random.uniform(0, 1, [n_components, N])

    for i in range(max_iter):
        Umatrix = Umatrix * Dmatrix.dot(Vmatrix.T) / (Umatrix.dot(Vmatrix.dot(Vmatrix.T)))
        Vmatrix = Vmatrix * (Umatrix.T.dot(Dmatrix) + alpha * Vmatrix.dot(W)) / (Umatrix.T.dot(Umatrix).dot(Vmatrix) + alpha * Vmatrix.dot(D))

    normalize = np.linalg.norm(Umatrix, axis=0)  # axis=1: row, axis=0: column
    # Umatrix = Umatrix.dot(np.diag(1.0 / normalize))  # could be ignored when clustering
    Vmatrix = np.diag(normalize).dot(Vmatrix)

    centroid, label, inertia = kmeans.k_means(Vmatrix.T, n_clusters=n_components)

    return centroid, label, inertia


###############################################################################
if __name__ == '__main__':
    print ('======================')
    data = np.random.rand(10, 8)
    centroid, label, inertia = gnmf(data, 3, alpha=0.1)
    print (centroid, label, inertia)
