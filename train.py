import re
import statistics
import sys

import numpy as np
import pandas as pd

from cnn import get_cnn_model_pred
from logistic_regression import get_lr_model_pred
from random_forest import get_rf_model_pred

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from scipy import stats as s
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


def get_mwe(mwe_file, idx_list):
    with open(mwe_file, 'r', errors='replace') as in_file:
        content = in_file.readlines()

        mwe_list = np.array([line.strip().split('\t')[0] for ind, line in enumerate(content) if ind in idx_list])

        mwe_dict = {}

        for mwe_ind, mwe in enumerate(mwe_list):
            if mwe not in mwe_dict.keys():
                mwe_dict[mwe] = np.array([mwe_ind])

            else:
                mwe_dict[mwe] = np.append(mwe_dict[mwe], mwe_ind)

        return mwe_list, mwe_dict


def load_transformer_embeddings_data(dataset_file, mwe_file):
    print(f'Reading file: {dataset_file.split("/")[-1]}')
    df = pd.read_csv(dataset_file, sep='\t', header=None)

    print('Generating embeddings list...')
    df[4] = df[0] + ',' + df[1] + ',' + df[2]

    embeddings_list = [elem.split(',') for elem in df[4].tolist()]

    correct_idx_list = np.array([ind for ind, sentence in enumerate(embeddings_list) if 'tensor(nan)' not in sentence])

    embeddings_list = [([float(re.findall(r"[-+]?\d*\.\d+|\d+", val)[0]) for val in sentence], label) for
                       sentence, label in zip(embeddings_list, df[3].tolist()) if 'tensor(nan)' not in sentence]

    mwe_list, mwe_dict = get_mwe(mwe_file, correct_idx_list)

    X = np.array([elem[0] for elem in embeddings_list])

    y = np.array([elem[1] for elem in embeddings_list])
    y = y.astype(int)

    indices = np.arange(X.shape[0])

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X,
                                                                                     y,
                                                                                     indices,
                                                                                     test_size=0.20,
                                                                                     random_state=42)

    return X_train, X_test, y_train, y_test, indices_train, indices_test, mwe_list, mwe_dict


def get_evaluation_report(y_true, y_pred):
    target_names = ['Incorrect MWE', 'Correct MWE']

    print(classification_report(y_true, y_pred, target_names=target_names))


def get_majority_voting(y_pred, mwe_dict, indices_test):
    y_majority_pred = np.array([0 for _ in y_pred])

    for pred_ind, prediction in enumerate(y_pred):
        mwe_ind = indices_test[pred_ind]
        for ind_set in mwe_dict.values():
            if mwe_ind in ind_set:
                # print(f'y_pred idx: {pred_ind}',
                # f'y_pred value: {prediction}',
                # f'mwe_ind: {mwe_ind}',
                # f'ind_set containing mwe_ind: {ind_set}',
                # sep = '\n')

                predictions = [y_pred[indices_test.tolist().index(label_ind)] for label_ind in ind_set if
                               label_ind in indices_test]
                # y_majority_pred[pred_ind] = statistics.mode(predictions)
                y_majority_pred[pred_ind] = int(s.mode(predictions)[0])
                break

    return y_majority_pred


def get_weighted_voting(y_pred, y_pred_max_probs, mwe_dict, indices_test):
    y_majority_pred = np.array([0 for _ in y_pred])

    for pred_ind, prediction in enumerate(y_pred):
        mwe_ind = indices_test[pred_ind]
        for ind_set in mwe_dict.values():
            if mwe_ind in ind_set:
                predictions_with_probs = [(y_pred[indices_test.tolist().index(label_ind)],
                                           y_pred_max_probs[indices_test.tolist().index(label_ind)]) for label_ind in
                                          ind_set if
                                          label_ind in indices_test]

                weights_per_class = []

                for class_id in range(2):
                    weights_per_class[class_id] = sum([elem[1] for elem in predictions_with_probs if elem[0] == class_id])

                y_majority_pred[pred_ind] = int(np.argmax(weights_per_class))

                break

    return y_majority_pred


def main(args):
    if 'transformer_embeddings' in args:
        dataset_filepath = 'sentences_containing_mwe_from_kgr10_group_0_embeddings_1_layers_incomplete_mwe_in_sent.tsv'
        mwe_filepath = 'sentences_containing_mwe_from_kgr10_group_0_mwe_list_incomplete_mwe_in_sent.tsv'

        X_train, X_test, y_train, y_test, indices_train, indices_test, mwe_list, mwe_dict = load_transformer_embeddings_data(
            dataset_filepath, mwe_filepath)

    else:
        dataset_filepath = 'mwe_dataset.npy'
        # dataset_filepath = 'mwe_dataset_cbow.npy'
        # dataset_filepath = 'mwe_dataset_domain_balanced.npy'  # domain-balanced dataset

        X_train, X_test, y_train, y_test = load_data(dataset_filepath)

    if 'smote' in args:
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)

    if 'pipeline' in args:
        over = SMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.5)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        X_train, y_train = pipeline.fit_resample(X_train, y_train)

    if 'borderline_smote' in args:
        oversample = BorderlineSMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)

    if 'svm_smote' in args:
        oversample = SVMSMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)

    if 'adasyn' in args:
        oversample = ADASYN()
        X_train, y_train = oversample.fit_resample(X_train, y_train)

    if 'cnn' in args:
        print(f'X_train shape: {X_train.shape}')
        X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], 1])
        X_test = np.reshape(X_test, [X_test.shape[0], X_train.shape[1], 1])
        y_train = one_hot(y_train, depth=2)

        if 'eval' in args:
            eval_only = True
            model_path = args[2]
        else:
            eval_only = False
            model_path = None

        y_pred_probs = get_cnn_model_pred(X_train, y_train, X_test,
                                          eval_only=eval_only,
                                          model_path=model_path,
                                          input_shape=(X_train.shape[1], 1))

        y_pred_max_probs = [max(probs) for probs in y_pred_probs]

        y_pred = [np.argmax(probs) for probs in y_pred_probs]

    elif 'lr' in args:
        y_pred, y_pred_probs = get_lr_model_pred(X_train, y_train, X_test)

        y_pred_max_probs = [max(probs) for probs in y_pred_probs]

    elif 'rf' in args:
        y_pred, y_pred_probs = get_rf_model_pred(X_train, y_train, X_test)

        y_pred_max_probs = [max(probs) for probs in y_pred_probs]

    if 'majority_voting' in args:
        y_pred = get_majority_voting(y_pred, mwe_dict, indices_test)

    if 'weighted_voting':
        y_pred = get_weighted_voting(y_pred, y_pred_max_probs, mwe_dict, indices_test)

    get_evaluation_report(y_test, y_pred)


if __name__ == '__main__':
    main(sys.argv[1:])
