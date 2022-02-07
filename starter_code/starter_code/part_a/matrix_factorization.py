from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # set up u_m and z_m for derivatives
    u_n = u[n]
    z_m = z[q]

    # compute the gradient descents
    u_n -= lr * (c - np.dot(u_n.T, z_m)) * (-1) * z_m
    z_m -= lr * (c - np.dot(u_n.T, z_m)) * (-1) * u_n

    # replace with the updated U and Z
    u[n] = u_n
    z[q] = z_m
    return u, z


def als(train_data, k, lr, num_iteration, val_data):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))
    train_loss = []
    val_loss = []

    for iter in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        # opted to do every 5000th iter because program was running too slow
        if iter % 5000 == 0:
            train_loss.append(squared_error_loss(train_data, u, z))
            val_loss.append(squared_error_loss(val_data, u, z))

    mat = np.dot(u, z.T)
    return mat, train_loss, val_loss


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    """
    # set up a list of different k to test
    k = [1, 3, 5, 7, 9, 12, 15, 20, 25, 30]
    # list to store the accuracy
    svd_acc = []
    for i in k:
        mat = svd_reconstruct(train_matrix, i)
        # accuracy
        acc = sparse_matrix_evaluate(val_data, mat)
        svd_acc.append(acc)
        print("For k = {0}, Validation Accuracy: {1}".format(i, acc))
    # best k
    max_k = k[svd_acc.index(max(svd_acc))]
    # reconstruct the matrix using the best k
    mat = svd_reconstruct(train_matrix, max_k)
    # validation and test accuracy
    svd_val_acc = sparse_matrix_evaluate(val_data, mat)
    svd_test_acc = sparse_matrix_evaluate(test_data, mat)
    print("Final validation accuracy with k* = {0} is {1}".format(max_k, svd_val_acc))
    print("Final test accuracy with k* = {0} is {1}".format(max_k, svd_test_acc))
    """
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    np.random.seed(seed=1000)

    lr = 0.01
    num_iter = 500000
    # set up a list of different k to test
    k = [1, 10, 30, 70, 100, 150, 200, 300, 500]
    # list to store the accuracy
    als_acc = []
    for i in k:
        mat, train_loss, val_loss = als(train_data, i, lr, num_iter, val_data)
        acc = sparse_matrix_evaluate(val_data, mat)
        als_acc.append(acc)
        print("For k = {0}, Validation Accuracy: {1}".format(i, acc))
    # best k
    max_k = k[als_acc.index(max(als_acc))]
    # reconstruct the matrix using the best k
    mat, train_loss, val_loss = als(train_data, max_k, lr, num_iter, val_data)
    # validation and test accuracy
    als_val_acc = sparse_matrix_evaluate(val_data, mat)
    als_test_acc = sparse_matrix_evaluate(test_data, mat)
    print("Final validation accuracy with k* = {0} is {1}".format(max_k, als_val_acc))
    print("Final test accuracy with k* = {0} is {1}".format(max_k, als_test_acc))

    iter = [x * 500 for x in range(len(train_loss))]
    plt.plot(iter, train_loss, label="training")
    plt.plot(iter, val_loss, label="validation")
    plt.ylabel("Squared-error-losses")
    plt.xlabel("Iterations")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
