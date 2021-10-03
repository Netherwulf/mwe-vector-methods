import sys

import numpy as np

from cnn import get_cnn_model_pred
from logistic_regression import get_lr_model_pred

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from tensorflow import one_hot


def load_data(dataset_file):
    dataset = np.load(dataset_file)

    X = np.array([elem[:900] for elem in dataset])

    y = np.array([elem[900] for elem in dataset])
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def get_evaluation_report(y_true, y_pred):
    target_names = ['Incorrect MWE', 'Correct MWE']

    print(classification_report(y_true, y_pred, target_names=target_names))


def main(args):
    dataset_filepath = 'mwe_dataset.npy'
    # dataset_filepath = 'mwe_dataset_domain_balanced.npy'  # domain-balanced dataset

    X_train, X_test, y_train, y_test = load_data(dataset_filepath)
    if 'cnn' in args:
        X_train = np.reshape(X_train, [X_train.shape[0], 900, 1])
        X_test = np.reshape(X_test, [X_test.shape[0], 900, 1])
        y_train = one_hot(y_train, depth=2)

        if 'eval' in args:
            eval_only = True
            model_path = args[2]
        else:
            eval_only = False
            model_path = None

        y_pred = get_cnn_model_pred(X_train, y_train, X_test,
                                    eval_only=eval_only,
                                    model_path=model_path)

        y_pred = [np.argmax(probs) for probs in y_pred]

    elif 'lr' in args:
        y_pred = get_lr_model_pred(X_train, y_train, X_test)

    get_evaluation_report(y_test, y_pred)


if __name__ == '__main__':
    main(sys.argv[1:])
