# -*- coding: utf-8 -*-

import numpy as np
from sklearn import cluster
from sklearn import metrics

from util import data
from util import alignment


# data: samples * features
def ncut(data, n_clusters=8, eigen_solver=None, random_state=None,
         n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1,
         kernel_params=None, n_jobs=1):

    label_predict = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver=eigen_solver,
                               random_state=random_state, n_init=n_init, gamma=gamma,
                               affinity=affinity, n_neighbors=n_neighbors, eigen_tol=eigen_tol,
                               assign_labels=assign_labels, degree=degree, coef0=coef0, kernel_params=kernel_params,
                               n_jobs=n_jobs).fit_predict(data)

    return label_predict


def predict2trueV2_labels(label_true, label_predict):
    one2one_true_predict, one2one_predict_true = alignment.one2one_lookup(label_true, label_predict)

    # mapping
    for i in range(np.array(label_predict).size):
        label_predict[i] = int(one2one_predict_true[label_predict[i]])

    return label_predict


###############################################################################
if __name__ == '__main__':
    print ('======================')
    #
    X, y = data.loader('coil')
    X = X.T
    y = np.array(y).flatten()
    print (X.shape, y.shape)

    #
    label_predict = ncut(X, n_clusters=np.unique(y).size)

    #
    label_predict = predict2trueV2_labels(y, label_predict)
    nmi = metrics.normalized_mutual_info_score(y, label_predict)
    acc = metrics.accuracy_score(y, label_predict)
    print ('nmi, acc:', nmi, acc)
