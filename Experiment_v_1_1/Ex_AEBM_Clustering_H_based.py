# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as scio
from sklearn import metrics

from baselines import kmeans
from configure import properties
from util import purity
from util import data
from util import alignment
from util import time
from util import logger
from util import bestMap
import acbm_main_entrance_data_without_paint as AEBM


datasets = {
    #'orl': 'orl',
    'umist': 'umist',#1
    #'olivetti': 'olivetti',
    #'yale': 'yale',
    #'yaleB': 'yaleB',
    #'usps': 'usps',
    #'mnist': 'mnist',
    #'alpha_digit': 'alpha_digit',
    #'fashion_mnist': 'fashion_mnist',
    #'coil': 'coil',
    #'ar': 'ar',
    #'pie': 'pie',
}

#datasets = {
#    'umist': 'umist',
#}

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
    k = nClusters[datasets[name]]
    w_a, s_a, v_a, flag_list, flag_aim_list, flag_w_list, flag_s_list, flag_v_list = AEBM.ACBM(k, corpus, name)
    h = w_a.T.dot(corpus)
    # clustering
    hh=h.T
    centroid, label_predict, inertia = kmeans.k_means(hh, n_clusters=nClusters[datasets[name]])


    # metrics: accuracy and nmi
    # label_predict = predict2true_labels(centroid, corpus, label_true, label_predict)
    nmi = metrics.normalized_mutual_info_score(label_true, label_predict)

    label_predict = bestMap.predict2trueV2_labels(label_true, label_predict)
    acc = metrics.accuracy_score(label_true, label_predict)
    purit=purity.get_purity(label_true, label_predict)

    return acc, nmi,  purit


if __name__ == '__main__':
    logs = logger.log("../" + 'model/aebm_kmeans/log.log', mode='w')
    logs.info('Kmeans starts:' + time.stamp())

    print (datasets, datasets.__len__())
    logs.info(datasets)

    print (nClusters, nClusters.__len__())
    logs.info(nClusters)

    # kmeans results
    Results_Kmeans_Clustering = {}
    for key in sorted(datasets.keys()):
        dataset_name = datasets.get(key)
        print(key, dataset_name)
        logs.info('======================================================')
        logs.info(key + ':' + dataset_name)

        repeatTimes = 10
        acc_all_results = np.zeros(repeatTimes)
        nmi_all_results = np.zeros(repeatTimes)
        purit_all_results = np.zeros(repeatTimes)
        for i in range(repeatTimes):
            acc, nmi, purit = acc_nmi_evaluation(dataset_name)
            acc_all_results[i] = acc
            nmi_all_results[i] = nmi
            purit_all_results[i] = purit
        print ('acc_all_results:', acc_all_results)
        print ('nmi_all_results:', nmi_all_results)

        logs.info('acc_all_results:')
        logs.info(acc_all_results)
        logs.info('nmi_all_results:')
        logs.info(nmi_all_results)

        Results_Kmeans_Clustering[dataset_name] = {'acc': np.mean(acc_all_results),
                                                   'std_acc': np.std(acc_all_results),
                                                   'nmi': np.mean(nmi_all_results),
                                                   'std_nmi': np.std(nmi_all_results),
                                                   'pur': np.mean(purit_all_results),
                                                   'std_pur': np.std(purit_all_results)
                                                   }
        print ('======================================================')
        logs.info('======================================================')

    print ('Results_Kmeans_Clustering:', Results_Kmeans_Clustering)
    logs.info('Results_Kmeans_Clustering:')
    logs.info(Results_Kmeans_Clustering)

    # save results to mat
    file_path = '../model'  + '/aebm_kmeans' +'/Results_Kmeans_Clustering.mat'
    scio.savemat(file_path, Results_Kmeans_Clustering)

    print ('Clustering experiments on data sets finished successfully!' + '@' + time.stamp())
    logs.info('Clustering experiments on data sets finished successfully!' + '@' + time.stamp())
