from utils import *
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.
    # assign variables for easier reference
    user_id, question_id, is_correct = data["user_id"], data["question_id"], data["is_correct"]

    for k in range(len(is_correct)):
        # theta for students ability, beta for questions difficulty
        theta_i = theta[user_id[k]]
        beta_j = beta[question_id[k]]
        # calculates the log-likelihood
        log_lklihood += is_correct[k] * (theta_i - beta_j) - np.log(1 + np.exp(theta_i - beta_j))

    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    # This is the helper function for alternating maximization
    # assign variables for easier reference
    user_id, question_id, is_correct = data["user_id"], data["question_id"], data["is_correct"]
    # theta and beta variables to store the values
    theta_return = np.zeros(theta.shape)
    beta_return = np.zeros(beta.shape)

    for k in range(len(is_correct)):
        # theta for students ability, beta for questions difficulty
        theta_i = theta[user_id[k]]
        beta_j = beta[question_id[k]]
        # calculate the derivatives wrt theta and beta
        diff = theta_i - beta_j
        sig = sigmoid(diff)
        theta_return[user_id[k]] += sig - is_correct[k]
        beta_return[question_id[k]] += is_correct[k] - sig
    # return the gradient descents, subtract for max (neg) log-likelihood
    theta -= lr * theta_return
    beta -= lr * beta_return

    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # random numbers to initialize theta and beta
    theta = np.random.rand(542)
    beta = np.random.rand(1774)

    val_acc_lst = []
    val_log_lst = []
    train_log_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

        train_log_lst.append(neg_lld)
        val_log_lst.append(neg_log_likelihood(val_data, theta=theta, beta=beta))

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_log_lst, val_log_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    theta, beta, val_acc_lst, train_log_lst, val_log_lst = irt(train_data, val_data, 0.01, 100)

    plt.plot(range(100), train_log_lst, label="training")
    plt.plot(range(100), val_log_lst, label="validation")
    plt.xlabel("Iterations")
    plt.ylabel("Negative log-likelihood")
    plt.legend()
    plt.show()

    plt.plot(range(100), val_acc_lst)
    plt.xlabel("Iterations")
    plt.ylabel("Validation accuracy")
    plt.show()

    theta, beta, test_acc_lst, train_log_lst, test_log_lst = irt(train_data, test_data, 0.01, 100)
    plt.plot(range(100), test_acc_lst)
    plt.xlabel("Iterations")
    plt.ylabel("Testing accuracy")
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # For (d)
    # sort theta, otherwise will have a really messy plot
    theta = np.sort(theta)
    for j in (100, 200, 300):
        beta_j = beta[j]
        prob = sigmoid(theta - beta_j)
        print(prob.shape)
        plt.plot(theta, prob, label="Question " + str(j))
    plt.ylabel("Probability of the correct response")
    plt.xlabel("Theta")
    plt.legend()
    plt.show()
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
