from sklearn.impute import KNNImputer
import numpy as np
from clustering import *
from matrix_factorization_weighted import *


############## ignore please, algorithm did not work out ##############################


def ensemble_wmf(data, train_data, k, lr, num_iter, val_data, epsilon, lamb):
    """ Ensemble the weighted matrix factorization algorithm using clustering

    :param data: A dictionary with key as the clustered questions, values in
    the format of the original data dictionary.
    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: int
    :param num_iter: int
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param epsilon: int
    :param lamb: int
    :return: list
    """
    mat_lst = []
    for key in data.keys():
        train_data = data[key]
        mat, train_loss, val_loss = als(train_data, k, lr, num_iter, val_data, epsilon, lamb)
        mat_lst.append(mat)
    return mat_lst


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k, add_indicator=True)
    # We use NaN-Euclidean distance measure.
    # Use transpose of the sparse matrix for item-based cf
    mat = nbrs.fit_transform(matrix.T).T
    return mat


def ensemble_knn(data, val_data, k):
    """ Ensemble the knn by item algorithm using clustering

    :param data: A dictionary with key as the clustered questions, values in
    the format of the original data dictionary.
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :return: list
    """
    mat = np.zeros((542, 1774))
    for mat in data:
        # computes matrix for items
        cal_mat = knn_impute_by_item(mat, val_data, k)
        cal_mat = cal_mat[:542, :]
        mat = mat + cal_mat
    mat = mat/542
    return mat


def avg_sparse_matrix(matrix_list, data_list):
    # Helper function to take the average of the matrices
    # initialize a new empty matrix
    mat = np.zeros((542, 1774))
    for index, key in enumerate(data_list.keys()):
        data = data_list[key]
        user = data["user_id"]
        question = data["question_id"]
        correct = data["is_correct"]
        mat = mat + matrix_list[index]
    mat = mat / len(data_list.keys())
    return mat


def form_sparse_matrix(data_list):
    mat_list = []
    for key in data_list.keys():
        # initialize a matrix for every cluster, filled with NAs
        mat = np.zeros((542, 1774))
        mat[:] = np.nan
        data = data_list[key]
        user = data["user_id"]
        question = data["question_id"]
        correct = data["is_correct"]
        for i in range(len(correct)):
            cur_user = user[i]
            cur_question = question[i]
            mat[cur_user][cur_question] = correct[i]
        mat_list.append(mat)
    return mat_list


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    question_meta_data = load_question_meta_csv("../data")

    np.random.seed(seed=1000)

    lr = 0.01
    num_iter = 500000
    epsilon = 1.9
    lamb = 0.01
    k = 7

    data = clustering(question_meta_data, train_data)
    # mat_lst = ensemble_wmf(data, train_data, k, lr, num_iter, val_data, epsilon, lamb)
    # mat = avg_sparse_matrix(mat_lst, data)
    # print(sparse_matrix_evaluate(val_data, mat))
    data_mat = form_sparse_matrix(data)
    print(ensemble_knn(data_mat, val_data, 5))


if __name__ == "__main__":
    main()

