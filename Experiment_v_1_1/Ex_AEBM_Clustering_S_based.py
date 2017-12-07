# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as scio
from sklearn import metrics

from baselines import spectralclustering
from configure import properties
from util import data
from util import alignment
from util import time
from util import logger
from util import purity
import acbm_main_entrance_data_without_paint as AEBM

datasets = {
    'orl': 'orl',
    'umist': 'umist',
    'olivetti': 'olivetti',
    'yale': 'yale',
    'yaleB': 'yaleB',
    'usps': 'usps',
    'mnist': 'mnist',
    'alpha_digit': 'alpha_digit',
    'fashion_mnist': 'fashion_mnist',
    'coil': 'coil',
    'ar': 'ar',
    'pie': 'pie',
}
'''
datasets = {
    'yaleB': 'yaleB',
}
'''
nClusters = {
    'orl': 40,
    'umist': 20,
    'olivetti': 40,
    'yale': 15,
    'yaleB': 38,
    'usps': 10,
    'mnist': 10,
    'alpha_digit': 36,
    'fashion_mnist': 10,
    'coil': 20,
    'ar': 10,
    'pie': 10,
}


def predict2true_labels(centroid, data, label_true, label_predict):  # data v.s. label_true = 1 : 1
    """
    Generally, clustering methods will automatically assign labels to each clustering centroid, but it's usually not
    consistent with the labels given by experts. The following function maps the predicted labels to expert labels.
    :param centroid:
    :param data:
    :param label_true:
    :param label_predict:
    :return:
    """
    label_true = np.array(label_true)

    row, col = data.shape
    # print 'row, col:', row, col

    # number of classes
    label_list = np.unique(label_true)
    # print 'label_list:', label_list

    # store center of data samples
    center_data = np.zeros(centroid.shape)  # data center by averaging the sample in one class
    for i in range(label_list.size):
        mark = label_list[i]
        places = np.where(label_true == mark)[0]
        # print 'places:', places

        vector = np.zeros((1, col))
        for j in range(places.size):
            vector = vector + data[places[j]]
        vector = vector / places.size
        center_data[i, :] = vector
        # print 'vector:', vector

    # distances between centroid (clustering results) and centers (data results)
    # print 'label_list.size:', label_list.size
    distance = np.zeros((label_list.size, label_list.size))
    for i in range(label_list.size):
        for j in range(label_list.size):
            distance[i, j] = np.linalg.norm(centroid[i, :] - center_data[j, :])

    # one2one lookup
    one2one = alignment.mapping(distance)
    # print 'one2one lookups:', one2one

    # mapping
    for i in range(np.array(label_predict).size):
        label_predict[i] = int(one2one[label_predict[i]])

    return label_predict


def predict2trueV2_labels(label_true, label_predict):
    one2one_true_predict, one2one_predict_true = alignment.one2one_lookup(label_true, label_predict)

    # mapping
    for i in range(np.array(label_predict).size):
        label_predict[i] = int(one2one_predict_true[label_predict[i]])

    return label_predict


def acc_nmi_evaluation(dataset_name):
    """
    :param dataset_name:
    :return:
    """

    # KMeans on data set according specific name, i.e. 'orl'
    name = dataset_name

    # load data and corresponding labels
    corpus, label_true = data.loader(datasets[name])

    #corpus = corpus.T  # features * samples --->  samples * features, i.e. each row corresponds to one sample
    label_true = np.array(label_true).flatten()
    ''' '''
    k=nClusters[name]
    w_a, s_a, v_a, flag_list, flag_aim_list, flag_w_list, flag_s_list, flag_v_list = AEBM.ACBM(k,corpus,name)
    # clustering
    label_predict = spectralclustering.ncut(corpus, n_clusters=nClusters[datasets[name]])

    # metrics: accuracy and nmi
    # label_predict = predict2true_labels(centroid, corpus, label_true, label_predict)
    label_predict = predict2trueV2_labels(label_true, label_predict)

    acc = metrics.accuracy_score(label_true, label_predict)
    nmi = metrics.normalized_mutual_info_score(label_true, label_predict)
    purit = purity.get_purity(label_true, label_predict)
    return acc, nmi, purit


if __name__ == '__main__':
    logs = logger.log(properties.base_path() + 'model/ncut/log.log', mode='w')
    logs.info('Ncut starts:' + time.stamp())

    print (datasets, datasets.__len__())
    logs.info(datasets)

    print (nClusters, nClusters.__len__())
    logs.info(nClusters)

    # kmeans results
    Results_NCut_Clustering = {}
    for key in sorted(datasets.keys()):
        dataset_name = datasets.get(key)
        print(key, dataset_name)
        logs.info('======================================================')
        logs.info(key + ':' + dataset_name)

        repeatTimes = 10
        acc_all_results = np.zeros(repeatTimes)
        nmi_all_results = np.zeros(repeatTimes)
        pur_all_results = np.zeros(repeatTimes)
        for i in range(repeatTimes):
            acc, nmi, pur = acc_nmi_evaluation(dataset_name)
            acc_all_results[i] = acc
            nmi_all_results[i] = nmi
            pur_all_results[i] = pur
        print ('acc_all_results:', acc_all_results)
        print ('nmi_all_results:', nmi_all_results)

        logs.info('acc_all_results:')
        logs.info(acc_all_results)
        logs.info('nmi_all_results:')
        logs.info(nmi_all_results)

        Results_NCut_Clustering[dataset_name] = {'acc': np.mean(acc_all_results),
                                                 'std_acc': np.std(acc_all_results),
                                                 'nmi': np.mean(nmi_all_results),
                                                 'std_nmi': np.std(nmi_all_results),
                                                 'pur': np.mean(pur_all_results),
                                                 'std_pur': np.std(pur_all_results)
                                                 }
        print ('======================================================')
        logs.info('======================================================')

    print ('Results_NCut_Clustering:', Results_NCut_Clustering)
    logs.info('Results_NCut_Clustering:')
    logs.info(Results_NCut_Clustering)

    # save results to mat
    file_path = properties.base_path() + 'model' + properties.separator() + 'ncut' + properties.separator() + 'Results_NCut_Clustering.mat'
    scio.savemat(file_path, Results_NCut_Clustering)

    print ('Clustering experiments on data sets finished successfully!' + '@' + time.stamp())
    logs.info('Clustering experiments on data sets finished successfully!' + '@' + time.stamp())
