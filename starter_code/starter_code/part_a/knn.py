from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    # print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    # Use transpose of the sparse matrix for item-based cf
    mat = nbrs.fit_transform(matrix.T)
    # Transpose back to fit the data dimension
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    # print("Validation Accuracy: {}".format(acc))
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    # Set the k values as required in handout
    k = [1, 6, 11, 16, 21, 26]

    # For user-based cf
    user_acc = []
    for i in k:
        acc = knn_impute_by_user(sparse_matrix, val_data, i)
        print("Accuracy on the validation data for user-based collaborative filtering with k = {0} is {1}".format(i, acc))
        user_acc.append(acc)
    # plot the acc against k
    plt.plot(k, user_acc)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.show()
    # k^*
    max_k = k[user_acc.index(max(user_acc))]
    user_test_acc = knn_impute_by_user(sparse_matrix, test_data, max_k)
    print("Final test accuracy for user-based collaborative filtering with k* = {0} is {1}"
          .format(max_k, user_test_acc))

    # For item-based cf
    item_acc = []
    for i in k:
        acc = knn_impute_by_item(sparse_matrix, val_data, i)
        print("Accuracy on the validation data for item-based collaborative filtering with k = {0} is {1}".format(i, acc))
        item_acc.append(acc)
    # plot the acc against k
    plt.plot(k, item_acc)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.show()
    # k^*
    max_k = k[item_acc.index(max(item_acc))]
    item_test_acc = knn_impute_by_item(sparse_matrix, test_data, max_k)
    print("Final test accuracy for item-based collaborative filtering with k* = {0} is {1}"
          .format(max_k, item_test_acc))


if __name__ == "__main__":
    main()
