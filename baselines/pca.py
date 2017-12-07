# -*- coding: utf-8 -*-

import numpy as np

from sklearn import decomposition
from baselines import kmeans


# data: samples * features by default
def pca(data, n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto',
        random_state=None):
    model = decomposition.PCA(n_components=n_components, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol,
                              iterated_power=iterated_power, random_state=random_state)

    newData = model.fit_transform(data)
    centroid, label, inertia = kmeans.k_means(newData, n_clusters=n_components)

    return centroid, label, inertia
