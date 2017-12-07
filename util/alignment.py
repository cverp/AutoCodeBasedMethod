# coding = 'utf-8'

import numpy as np


# this kind of mapping is based on distance matrix. (predict-true mapping)
def mapping(_2DArray):  # _2DArray must be square
    # size of _2DArray
    row, col = _2DArray.shape
    # print row, col

    # extend, and init
    extend_2DArray = np.zeros((row + 1, col + 1))

    extend_2DArray[0, 0] = -1  # placeholder, no use
    for i in range(row):
        extend_2DArray[0, i + 1] = i

    for j in range(col):
        extend_2DArray[j + 1, 0] = j

    for i in range(row):
        for j in range(col):
            extend_2DArray[i + 1, j + 1] = _2DArray[i, j]

    # print extend_2DArray

    # mapping
    one2one = np.zeros(row)

    #
    for i in range(row):
        index_row, index_column = np.where(extend_2DArray == np.min(extend_2DArray[1:row, 1:col]))

        map_row = extend_2DArray[index_row[0], 0]  #
        map_col = extend_2DArray[0, index_column[0]]  #
        # print 'row:col', map_row, map_col
        one2one[int(map_row)] = int(map_col)

        #
        extend_2DArray = np.delete(extend_2DArray, index_row[0],
                                   0)  # 0(3rd), delete row; index_row[0] + 1: delete the corresonding row
        extend_2DArray = np.delete(extend_2DArray, index_column[0],
                                   1)  # 1(3rd), delete column; index_column[0] + 1: delete the corresonding column

    return one2one


# map label_true to label_predict, and vice wise.
# this kind of mapping is based on label_true and label_predict
def one2one_lookup(label_true, label_predict):
    # unique classes and size
    label_list = np.unique(label_true)
    label_size = label_list.size

    # true-predict pairwise mapping and init
    true_predict = np.zeros((label_size + 1, label_size + 1))
    true_predict[0, 0] = -1
    for i in range(label_size):
        true_predict[0, i + 1] = i
        true_predict[i + 1, 0] = i

    for i in range(label_size):
        mark = label_list[i];
        places = np.where(label_true == mark)[0]
        # print 'places', places

        for j in range(places.size):
            for k in range(label_size):
                if label_predict[places[j]] == k:
                    # print 'label_size, j, k', label_size, j, k
                    true_predict[i + 1, k + 1] = true_predict[i + 1, k + 1] + 1.0 / places.size
                    # true_predict[i + 1, k + 1] = true_predict[i + 1, k + 1] + 1.0 / places.size

    # print 'true_predict', true_predict
    # add a number in avoiding np.max operation (exclude the 0-th row and column)
    for i in range(label_size):
        for j in range(label_size):
            true_predict[i + 1, j + 1] = true_predict[i + 1, j + 1] + label_size

    # print 'true_predict', true_predict

    # one2one lookup
    one2one_true_predict = np.ndarray((label_size,), int)
    one2one_predict_true = np.ndarray((label_size,), int)

    for i in range(label_size):
        index_row, index_column = np.where(true_predict == np.max(true_predict[1:, 1:]))
        # print 'true_predict[1:, 1:])', true_predict[1:, 1:]
        # print 'index_row, index_column', index_row, index_column

        map_row = true_predict[index_row[0], 0]  #
        map_col = true_predict[0, index_column[0]]  #
        # print 'row:col', map_row, map_col
        one2one_true_predict[int(map_row)] = int(map_col)
        one2one_predict_true[int(map_col)] = int(map_row)

        #
        true_predict = np.delete(true_predict, index_row[0],
                                 0)  # 0(3rd), delete row; index_row[0] + 1: delete the corresonding row
        true_predict = np.delete(true_predict, index_column[0],
                                 1)  # 1(3rd), delete column; index_column[0] + 1: delete the corresonding column

        # print 'true_predict, shape', true_predict, true_predict.shape

    return one2one_true_predict, one2one_predict_true


###############################################################################
if __name__ == '__main__':
    print 'start...'

    _2DArray = np.random.rand(3, 3)
    one2one = mapping(_2DArray)
    print one2one
    print int(one2one[0])

    label_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    label_predict = np.array([1, 1, 1, 1, 2, 3, 3, 3, 0, 0, 0, 0, 2, 2, 2, 0])
    print one2one_lookup(label_true, label_predict)
