# -*- coding: utf-8 -*-

import numpy as np
from baselines import kmeans


# data: samples * features by default
def nmf(Dmatrix, n_components, max_iter=200):
    """
    self-defined NMF with mu algorithm
    :param Dmatrix:
    :param n_components:
    :param max_iter:
    :return:
    """
    Dmatrix = Dmatrix.T

    (M, N) = Dmatrix.shape
    Umatrix = np.random.uniform(0, 1, [M, n_components])
    Vmatrix = np.random.uniform(0, 1, [n_components, N])

    for i in range(max_iter):
        Umatrix = Umatrix * Dmatrix.dot(Vmatrix.T) / (Umatrix.dot(Vmatrix.dot(Vmatrix.T)))
        Vmatrix = Vmatrix * (Umatrix.T.dot(Dmatrix)) / (Umatrix.T.dot(Umatrix).dot(Vmatrix))

    normalize = np.linalg.norm(Umatrix, axis=0)
    # Umatrix = Umatrix.dot(np.diag(1.0 / normalize))  # could be ignored when clustering
    Vmatrix = np.diag(normalize).dot(Vmatrix)

    centroid, label, inertia = kmeans.k_means(Vmatrix.T, n_clusters=n_components)

    return centroid, label, inertia


###############################################################################
if __name__ == '__main__':
    print ('======================')
    data = np.random.rand(10, 10)
    centroid, label, inertia = nmf(data, 3)
    print (centroid, label, inertia)
