# coding = 'utf-8'

import numpy as np
import os
import matplotlib.pyplot as plt
import time
import scipy.io as sio

from configure import properties
from sklearn import preprocessing


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

    print dataset + ' ' + 'loaded successfully!'

    return data, label


###############################################################################
# load image data set: ORL
def load_image_orl():
    #path = properties.parse_args("global", "project_path") + "data/" + "orl/" + "orl(28-23).mat"
    path = "../" + "data/" + "orl/" + "orl(28-23).mat"
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
    #path = properties.parse_args("global", "project_path") + "data/" + "umist/" + "umist(28-23).mat"
    path = "../" + "data/" + "umist/" + "umist(28-23).mat"
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
    #path = properties.parse_args("global", "project_path") + "data/" + "coil/" + "coil(32-32).mat"
    path = "../" + "data/" + "coil/" + "coil(32-32).mat"
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
    #path = properties.parse_args("global", "project_path") + "data/" + "usps/" + "usps(16-16).mat"
    path ="../"+ "data/" + "usps/" + "usps(16-16).mat"
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
    #path = properties.parse_args("global", "project_path") + "data/" + "olivetti/" + "olivetti(32-32).mat"
    path = "../" + "data/" + "olivetti/" + "olivetti(32-32).mat"

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
    #path = properties.parse_args("global", "project_path") + "data/" + "yale/" + "yale(32-32).mat"
    path = "../" + "data/" + "yale/" + "yale(32-32).mat"

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
    #path = properties.parse_args("global", "project_path") + "data/" + "mnist/" + "mnist(28-28).mat"
    path = "../" + "data/" + "mnist/" + "mnist(28-28).mat"

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
    #path = properties.parse_args("global", "project_path") + "data/" + "fashion_mnist/" + "fashion_mnist(28-28).mat"
    path = "../" + "data/" + "fashion_mnist/" + "fashion_mnist(28-28).mat"

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
    #path = properties.parse_args("global", "project_path") + "data/" + "alpha_digit/" + "alpha_digit(20-16).mat"
    path = "../" + "data/" + "alpha_digit/" + "alpha_digit(20-16).mat"
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
    #path = properties.parse_args("global", "project_path") + "data/" + "yaleB/" + "yaleB(32-32).mat"
    path = "../" + "data/" + "yaleB/" + "yaleB(32-32).mat"
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
    #path = properties.parse_args("global", "project_path") + "data/" + "ar10/" + "ar10.mat"
    path = "../" + "data/" + "ar10/" + "ar10.mat"
    alpha_digit = sio.loadmat(path)
    data = alpha_digit['ar']
    label = alpha_digit['label']

    data = preprocessing.normalize(data, axis=0)  # axis=1: row, axis=0: column

    # print('column:', data[:, 0])
    # print('sum:', sum(data[:, 11] * data[:, 11]))

    print("ar10 data.shape", data.shape)
    print("ar10 label.shape", label.shape)

    return data, label


###############################################################################
# load image data set: PIE10
def load_image_pie():
    #path = properties.parse_args("global", "project_path") + "data/" + "pie10/" + "pie10.mat"
    path = "../" + "data/" + "pie10/" + "pie10.mat"
    alpha_digit = sio.loadmat(path)
    data = alpha_digit['pie']
    label = alpha_digit['label']

    data = preprocessing.normalize(data, axis=0)  # axis=1: row, axis=0: column

    # print('column:', data[:, 0])
    # print('sum:', sum(data[:, 11] * data[:, 11]))

    print("PIE10 data.shape", data.shape)
    print("PIE10 label.shape", label.shape)

    return data, label


###############################################################################
if __name__ == '__main__':
    print loader(dataset='alpha_digit')


