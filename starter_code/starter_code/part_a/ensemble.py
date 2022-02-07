import random

import numpy as np
from sklearn.impute import KNNImputer
from scipy.sparse import csc_matrix
from utils import *


def sampling(matrix):
    """ Select random samples with replacement with the original train
    sample dimension.

    :param matrix: 2D sparse matrix
    :return: list
    """
    # choose random items with replacement
    n = matrix.shape[1]
    mat_lst = []
    for i in range(3):
        index = np.random.choice(n, size=n, replace=True)
        mat = matrix[:, index]
        mat_lst.append(mat)

    return mat_lst


def knn_impute_by_item(matrix, valid_data):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :return: float
    """
    res = np.zeros((542, 1774))
    for mat in matrix:
        k = np.random.randint(10, high=20, size=1)[0]
        nbrs = KNNImputer(n_neighbors=k)
        # We use NaN-Euclidean distance measure.
        # Use transpose of the sparse matrix for item-based cf
        mat = nbrs.fit_transform(mat.T).T
        res = res + mat
    res = res/3
    return res


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # set up ensemble
    matrix = sampling(train_matrix)
    mat = knn_impute_by_item(matrix, val_data)

    # record accuracy
    vaL_acc = sparse_matrix_evaluate(val_data, mat)
    print("Final test accuracy for bagging ensemble is {}".format(vaL_acc))
    test_acc = sparse_matrix_evaluate(test_data, mat)
    print("Final test accuracy for bagging ensemble is {}".format(test_acc))


if __name__ == "__main__":
    main()