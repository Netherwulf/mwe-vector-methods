import sys

import numpy as np

from logistic_regression import create_lr_model, train_lr_model, get_lr_model_predictions
from cnn import create_cnn_model, train_cnn_model, get_cnn_model_predictions

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


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
    # dataset_filepath = 'mwe_dataset.npy'
    dataset_filepath = 'mwe_dataset_domain_balanced.npy'

    X_train, X_test, y_train, y_test = load_data(dataset_filepath)

    lr_model = create_lr_model()

    lr_model = train_lr_model(lr_model, X_train, y_train)

    y_pred = get_lr_model_predictions(lr_model, X_test)

    get_evaluation_report(y_test, y_pred)


if __name__ == '__main__':
    main(sys.argv[1:])
