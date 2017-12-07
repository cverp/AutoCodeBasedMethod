# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 22:11:38 2017

@author: tzb
"""
import numpy as np

"""
Purity is defined as follows:
       Purity = (1/n)* sum(max(n_{ij}))
where n_{ij} is the number of data points in the j-th cluster that belong to the i-th class.
Purity measures the extent to which each cluster contains data points from primarily one class.
"""


def get_purity(y, predy):  # inputs: label_true, label_predict
    """
    compute the purity of clustering
    
    Arguments:
    y -- the true label of samples
    ypred -- the predicted label of samples
    predy -- 
    
    Returns:
    purity
    """

    if len(y) != len(predy):
        assert ('y与predy长度不同')
    n = len(y)
    y = get_classes(y)
    predy = get_classes(predy)
    totle = 0
    for key in predy:
        pred_cluster = predy[key]
        mx = 0
        for key1 in y:
            cluster = y[key1]
            mx = max(mx, len(set(cluster).intersection(set(pred_cluster))))
        totle = totle + mx

    purity = 1.0 * totle / n
    return purity


def get_classes(y):
    labels = np.unique(y)
    result = {}
    for label in labels:
        result[label] = []
        for i in range(len(y)):
            if y[i] == label:
                result[label].append(i)
    return result


if __name__ == '__main__':
    y = np.array([1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 1, 3, 3, 3])
    predy = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    print(get_purity(y, predy))
    y = [1, 2]
    predy = [1, 1]
    print(get_purity(y, predy))
    print(get_purity(predy, y))
