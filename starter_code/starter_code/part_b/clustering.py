import numpy as np
import pandas as pd
import collections
from utils import *


def _load_csv(path):
    # A helper function to load the csv file.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "question_id": [],
        "subject_id": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))
                data["subject_id"].append(row[1])
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def load_question_meta_csv(root_dir="/data"):
    """ Load the question meta data as a dictionary.

    :param root_dir: str
    :return: A dictionary {user_id: list, question_id: list, is_correct: list}
        WHERE
        question_id: a list of question id.
        subject_id: a list of subject ids.
    """
    path = os.path.join(root_dir, "question_meta.csv")
    return _load_csv(path)


def clustering(meta_data, data):
    """ Separate the data into clustering according to meta_data.

    :param meta_data: A dictionary {question_id: list, subject_id: string}
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :return: A dictionary with key as the clustered questions, values in
    the format of the original data dictionary.
    """
    # group question_id by subject_id
    meta_df = pd.DataFrame(meta_data)
    meta_data = meta_df.groupby("subject_id", as_index=False, sort=False)

    subject_lst = meta_data.agg(lambda x: tuple(x))["subject_id"].tolist()
    question_lst = meta_data.agg(lambda x: tuple(x))["question_id"].tolist()

    # create new train_data based on this clustering
    user = data["user_id"]
    question = data["question_id"]
    correct = data["is_correct"]

    d = {}

    # rearrange into the original data format
    for q in range(len(question_lst)):
        dict = {"user_id": [], "question_id": [], "is_correct": []}
        for i in range(len(user)):
            if question[i] in question_lst[q]:
                dict["user_id"].append(user[i])
                dict["question_id"].append(question[i])
                dict["is_correct"].append(correct[i])
                d[question_lst[q]] = dict

    return d


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    question_meta_data = load_question_meta_csv("../data")

    d = clustering(question_meta_data, train_data)


if __name__ == "__main__":
    main()
