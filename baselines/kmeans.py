# -*- coding: utf-8 -*-
from sklearn import cluster

from util import data
from util import alignment
from sklearn import metrics

import numpy as np
import scipy.io as scio


def k_means(data, n_clusters, init='k-means++', precompute_distances='auto', n_init=10, max_iter=300,
           verbose=False, tol=0.0001, random_state=None,
           copy_x=True, n_jobs=1, algorithm='auto', return_n_iter=False):

    (centroid, label, inertia) = cluster.k_means(data, n_clusters, init=init,
                                                 precompute_distances=precompute_distances, n_init=n_init, max_iter=max_iter,
                                                 verbose=verbose, tol=tol, random_state=random_state,
                                                 copy_x=copy_x, n_jobs=n_jobs, algorithm=algorithm, return_n_iter=return_n_iter)

    return centroid, label, inertia


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
    centroid, label_predict, inertia = k_means(X, n_clusters=np.unique(y).size)

    # scio.savemat('results.mat', {'label_true': y, 'label_predict': label_predict})

    #
    label_predict = predict2trueV2_labels(y, label_predict)
    nmi = metrics.normalized_mutual_info_score(y, label_predict)
    acc = metrics.accuracy_score(y, label_predict)
    print ('nmi, acc:', nmi, acc)
