import sys

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_data(dataset_file):
    dataset = np.load(dataset_file)

    X = np.array([elem[: 900] for elem in dataset])

    y = np.array([elem[900] for elem in dataset])
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    return X_train, X_test, y_train, y_test


def create_model():

    return LogisticRegression(random_state=0, max_iter=2000, verbose=1)


def train(model, X, y):
    model.fit(X, y)

    return model


def get_test_predictions(model, X_test):

    return model.predict(X_test)


def get_evaluation_report(y_true, y_pred):
    target_names = ['Incorrect MWE', 'Correct MWE']

    print(classification_report(y_true, y_pred, target_names=target_names))


def main(args):
    dataset_filepath = 'mwe_dataset.npy'

    X_train, X_test, y_train, y_test = load_data(dataset_filepath)

    lr_model = create_model()

    lr_model = train(lr_model, X_train, y_train)

    y_pred = get_test_predictions(lr_model, X_test)

    get_evaluation_report(y_test, y_pred)


if __name__ == '__main__':
    main(sys.argv[1:])
