from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

import numpy as np


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


def update_u_z(train_data, lr, u, z, epsilon, lamb):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :param epsilon: int
    :param lamb: int
    :return: (u, z)
    """
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # set up u_n, z_m and W_um for derivatives
    u_n = u[n]
    z_m = z[q]
    w_um = 1 + np.log(1 + c * 10**epsilon)

    # compute the weighted gradient descents
    u_n -= lr * w_um * (c - np.dot(u_n.T, z_m)) * (-1) * z_m.T + lamb * u_n.T
    z_m -= lr * w_um * (c - np.dot(u_n.T, z_m)) * (-1) * u_n.T + lamb * z_m.T

    # replace with the updated U and Z
    u[n] = u_n
    z[q] = z_m

    return u, z


def als(train_data, k, lr, num_iteration, val_data, epsilon, lamb):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param epsilon: int
    :param lamb: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(542, k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(1774, k))
    train_loss = []
    val_loss = []

    for iter in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z, epsilon, lamb)
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
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    np.random.seed(seed=1000)

    lr = 0.001
    num_iter = 500000
    epsilon = 1.9
    lamb = 0.01
    # set up a list of different k to test
    k = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # list to store the accuracy
    als_acc = []
    for i in k:
        mat, train_loss, val_loss = als(train_data, i, lr, num_iter, val_data, epsilon, lamb)
        acc = sparse_matrix_evaluate(val_data, mat)
        als_acc.append(acc)
        print("For k = {0}, Validation Accuracy: {1}".format(i, acc))
    # best k
    max_k = k[als_acc.index(max(als_acc))]
    # reconstruct the matrix using the best k
    mat, train_loss, val_loss = als(train_data, max_k, lr, num_iter, val_data, epsilon, lamb)
    # validation and test accuracy
    als_val_acc = sparse_matrix_evaluate(val_data, mat)
    als_test_acc = sparse_matrix_evaluate(test_data, mat)
    print("Final validation accuracy with k* = {0} is {1}".format(max_k, als_val_acc))
    print("Final test accuracy with k* = {0} is {1}".format(max_k, als_test_acc))
    print(train_loss)

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
