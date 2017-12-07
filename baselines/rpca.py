# -*- coding: utf-8 -*-

import numpy as np
from baselines import kmeans

from configure import properties
from sklearn import preprocessing
import scipy.io as sio


# (1). rpca_data is obtained from inexact_alm_rpca file (matlab code) (2). since python code is difficult to call
# maltab function, we decide to run maltab code separately for low-rank recovery from original data. (3) the matlab
# is collected in baselines directory. (4) the rpca_data is collected in rpca directory (5)data format: samples *
# features by default (6) Reference: http://blog.csdn.net/tiandijun/article/details/44917237 (7) inexact ALM - MATLAB
def rpca(rpca_data, n_clusters=None):
    centroid, label, inertia = kmeans.k_means(rpca_data, n_clusters=n_clusters)
    return centroid, label, inertia


###############################################################################
def loader(dataset='orl'):
    if dataset == 'orl':
        data, label = load_image_orl()
    elif dataset == 'umist':
        data, label = load_image_umist()
    elif dataset == 'coil':
        data, label = load_image_coil()
    elif dataset == 'usps':
        data, label = load_image_usps()
    elif dataset == 'olivetti':
        data, label = load_image_olivetti()
    elif dataset == 'yale':
        data, label = load_image_yale()
    elif dataset == 'mnist':
        data, label = load_image_mnist()
    elif dataset == 'fashion_mnist':
        data, label = load_image_fashion_mnist()
    elif dataset == 'alpha_digit':
        data, label = load_image_alpha_digit()
    elif dataset == 'yaleB':
        data, label = load_image_yaleB()
    elif dataset == 'ar':
        data, label = load_image_ar()
    elif dataset == 'pie':
        data, label = load_image_pie()
    else:
        print("others")
        return

    print (dataset + ' ' + 'loaded successfully!')

    return data, label


###############################################################################
# load image data set: ORL
def load_image_orl():
    path = properties.parse_args("global", "project_path") + "rpca_data/" + "orl(28-23).mat"
    orl = sio.loadmat(path)
    data = orl['orl']
    label = orl['label']

    data = preprocessing.normalize(data, axis=0)  # axis=1: row, axis=0: column

    print("orl data.shape", data.shape)
    print("orl label.shape", label.shape)

    return data, label


###############################################################################
# load image data set: UMIST
def load_image_umist():
    path = properties.parse_args("global", "project_path") + "rpca_data/" + "umist(28-23).mat"
    umist = sio.loadmat(path)
    data = umist['umist']
    label = umist['label']

    data = preprocessing.normalize(data, axis=0)  # axis=1: row, axis=0: column

    print("umist data.shape", data.shape)
    print("umist label.shape", label.shape)

    return data, label


###############################################################################
# load image data set: COIL
def load_image_coil():
    path = properties.parse_args("global", "project_path") + "rpca_data/" + "coil(32-32).mat"
    coil = sio.loadmat(path)
    data = coil['coil']
    label = coil['label']

    data = preprocessing.normalize(data, axis=0)  # axis=1: row, axis=0: column

    print("coil data.shape", data.shape)
    print("coil label.shape", label.shape)

    # print('column:', data[:, 0])
    # print('sum:', sum(data[:, 7] * data[:, 7]))

    return data, label


###############################################################################
# load image data set: USPS
def load_image_usps():
    path = properties.parse_args("global", "project_path") + "rpca_data/" + "usps(16-16).mat"
    usps = sio.loadmat(path)
    data = usps['usps']
    label = usps['label']

    data = preprocessing.normalize(data, axis=0)  # axis=1: row, axis=0: column

    # print('column:', data[:, 0])
    # print('sum:', sum(data[:, 9] * data[:, 9]))

    print("usps data.shape", data.shape)
    print("usps label.shape", label.shape)

    return data, label


###############################################################################
# load image data set: Olivetti
def load_image_olivetti():
    path = properties.parse_args("global", "project_path") + "rpca_data/" + "olivetti(32-32).mat"
    olivetti = sio.loadmat(path)
    data = olivetti['olivetti']
    label = olivetti['label']

    data = preprocessing.normalize(data, axis=0)  # axis=1: row, axis=0: column

    # print('column:', data[:, 0])
    # print('sum:', sum(data[:, 9] * data[:, 9]))

    print("olivetti data.shape", data.shape)
    print("olivetti label.shape", label.shape)

    return data, label


###############################################################################
# load image data set: Yale
def load_image_yale():
    path = properties.parse_args("global", "project_path") + "rpca_data/" + "yale(32-32).mat"
    yale = sio.loadmat(path)
    data = yale['yale']
    label = yale['label']

    data = preprocessing.normalize(data, axis=0)  # axis=1: row, axis=0: column

    # print('column:', data[:, 0])
    # print('sum:', sum(data[:, 9] * data[:, 9]))

    print("yale data.shape", data.shape)
    print("yale label.shape", label.shape)

    return data, label


###############################################################################
# load image data set: MNIST
def load_image_mnist():
    path = properties.parse_args("global", "project_path") + "rpca_data/" + "mnist(28-28).mat"
    mnist = sio.loadmat(path)
    data = mnist['mnist']
    label = mnist['label']

    data = preprocessing.normalize(data, axis=0)  # axis=1: row, axis=0: column

    # print('column:', data[:, 0])
    # print('sum:', sum(data[:, 9] * data[:, 9]))

    print("mnist data.shape", data.shape)
    print("mnist label.shape", label.shape)

    return data, label


###############################################################################
# load image data set: FASHION_MNIST
def load_image_fashion_mnist():
    path = properties.parse_args("global", "project_path") + "rpca_data/" + "fashion_mnist(28-28).mat"
    fashion_mnist = sio.loadmat(path)
    data = fashion_mnist['fashion_mnist']
    label = fashion_mnist['label']

    data = preprocessing.normalize(data, axis=0)  # axis=1: row, axis=0: column

    # print('column:', data[:, 0])
    # print('sum:', sum(data[:, 9] * data[:, 9]))

    print("fashion_mnist data.shape", data.shape)
    print("fashion_mnist label.shape", label.shape)

    return data, label


###############################################################################
# load image data set: ALPHA_DIGITS
def load_image_alpha_digit():
    path = properties.parse_args("global", "project_path") + "rpca_data/" + "alpha_digit(20-16).mat"
    alpha_digit = sio.loadmat(path)
    data = alpha_digit['alphadigit']
    label = alpha_digit['label']

    data = preprocessing.normalize(data, axis=0)  # axis=1: row, axis=0: column

    # print('column:', data[:, 0])
    # print('sum:', sum(data[:, 11] * data[:, 11]))

    print("alpha_digit data.shape", data.shape)
    print("alpha_digit label.shape", label.shape)

    return data, label


###############################################################################
# load image data set: yaleB
def load_image_yaleB():
    path = properties.parse_args("global", "project_path") + "rpca_data/" + "yaleB(32-32).mat"
    alpha_digit = sio.loadmat(path)
    data = alpha_digit['yaleB']
    label = alpha_digit['label']

    data = preprocessing.normalize(data, axis=0)  # axis=1: row, axis=0: column

    # print('column:', data[:, 0])
    # print('sum:', sum(data[:, 11] * data[:, 11]))

    print("yaleB data.shape", data.shape)
    print("yaleB label.shape", label.shape)

    return data, label


###############################################################################
# load image data set: AR10
def load_image_ar():
    path = properties.parse_args("global", "project_path") + "rpca_data/" + "ar10.mat"
    alpha_digit = sio.loadmat(path)
    data = alpha_digit['ar']
    label = alpha_digit['label']

    data = preprocessing.normalize(data, axis=0)  # axis=1: row, axis=0: column

    # print('column:', data[:, 0])
    # print('sum:', sum(data[:, 11] * data[:, 11]))

    print("AR10 data.shape", data.shape)
    print("AR10 label.shape", label.shape)

    return data, label


###############################################################################
# load image data set: PIE10
def load_image_pie():
    path = properties.parse_args("global", "project_path") + "rpca_data/" + "pie10.mat"
    alpha_digit = sio.loadmat(path)
    data = alpha_digit['pie']
    label = alpha_digit['label']

    data = preprocessing.normalize(data, axis=0)  # axis=1: row, axis=0: column

    # print('column:', data[:, 0])
    # print('sum:', sum(data[:, 11] * data[:, 11]))

    print("PIE10 data.shape", data.shape)
    print("PIE10 label.shape", label.shape)

    return data, label
