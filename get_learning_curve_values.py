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

        mwe_metadata = np.array([line.strip().split('\t') for ind, line in enumerate(content) if ind in idx_list])

        mwe_dict = {}

        for mwe_ind, mwe in enumerate(mwe_list):
            if mwe not in mwe_dict.keys():
                mwe_dict[mwe] = np.array([mwe_ind])

            else:
                mwe_dict[mwe] = np.append(mwe_dict[mwe], mwe_ind)

        return mwe_list, mwe_dict, mwe_metadata


def load_transformer_embeddings_data(dataset_file, mwe_file):
    print(f'Reading file: {dataset_file.split("/")[-1]}')
    df = pd.read_csv(dataset_file, sep='\t', header=None)

    print('Generating embeddings list...')
    df[4] = df[0] + ',' + df[1] + ',' + df[2]

    embeddings_list = [elem.split(',') for elem in df[4].tolist()]

    correct_idx_list = np.array([ind for ind, sentence in enumerate(embeddings_list) if 'tensor(nan)' not in sentence])

    embeddings_list = [([float(re.findall(r"[-+]?\d*\.\d+|\d+", val)[0]) for val in sentence], label) for
                       sentence, label in zip(embeddings_list, df[3].tolist()) if 'tensor(nan)' not in sentence]

    mwe_list, mwe_dict, mwe_metadata = get_mwe(mwe_file, correct_idx_list)

    X = np.array([elem[0] for elem in embeddings_list])

    y = np.array([elem[1] for elem in embeddings_list])
    y = y.astype(int)

    indices = np.arange(X.shape[0])

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X,
                                                                                     y,
                                                                                     indices,
                                                                                     test_size=0.20,
                                                                                     random_state=42)

    return X_train, X_test, y_train, y_test, indices_train, indices_test, mwe_list, mwe_dict, mwe_metadata


def create_empty_file(filepath):
    with open(filepath, 'w') as f:
        column_names_line = '\t'.join(['mwe', 'first_word_index', 'first_word_orth', 'first_word_lemma', 'sentence',
                                       'is_correct', 'model_prediction'])
        f.write(f"{column_names_line}\n")
        pass


def write_line_to_file(filepath, line):
    with open(filepath, 'a') as f:
        f.write(f'{line}\n')


def write_prediction_result_to_file(filepath, mwe_ind, mwe_metadata, prediction):
    sample_metadata = mwe_metadata[mwe_ind]

    mwe = sample_metadata[0]
    is_correct = sample_metadata[1]
    first_word_idx = sample_metadata[2]
    first_word_orth = sample_metadata[3]
    first_word_lemma = sample_metadata[4]
    sentence = sample_metadata[5]

    sample_description = '\t'.join([str(mwe), str(first_word_idx),
                                    str(first_word_orth), str(first_word_lemma),
                                    str(sentence), str(is_correct),
                                    str(prediction)])

    write_line_to_file(filepath, sample_description)


def get_evaluation_report(y_true, y_pred, write_results, indices_test, mwe_metadata, filepath):
    target_names = ['Incorrect MWE', 'Correct MWE']

    if write_results:
        for pred_ind, prediction in enumerate(y_pred):
            mwe_ind = indices_test[pred_ind]

            write_prediction_result_to_file(filepath, mwe_ind, mwe_metadata, prediction)

    print(classification_report(y_true, y_pred, target_names=target_names))


def get_majority_voting(y_pred, mwe_dict, indices_test, mwe_metadata, filepath):
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

                final_prediction = int(s.mode(predictions)[0])

                y_majority_pred[pred_ind] = final_prediction

                write_prediction_result_to_file(filepath, mwe_ind, mwe_metadata, final_prediction)

                break

    return y_majority_pred


def get_weighted_voting(y_pred, y_pred_max_probs, mwe_dict, indices_test, mwe_metadata, filepath):
    y_majority_pred = np.array([0 for _ in y_pred])

    for pred_ind, prediction in enumerate(y_pred):
        mwe_ind = indices_test[pred_ind]
        for ind_set in mwe_dict.values():
            if mwe_ind in ind_set:
                predictions_with_probs = [(y_pred[indices_test.tolist().index(label_ind)],
                                           y_pred_max_probs[indices_test.tolist().index(label_ind)]) for label_ind in
                                          ind_set if label_ind in indices_test]

                weights_per_class = [0.0 for _ in range(2)]

                for class_id in range(len(weights_per_class)):
                    weights_per_class[class_id] = sum(
                        [elem[1] for elem in predictions_with_probs if elem[0] == class_id])

                final_prediction = int(np.argmax(weights_per_class))

                y_majority_pred[pred_ind] = final_prediction

                write_prediction_result_to_file(filepath, mwe_ind, mwe_metadata, final_prediction)

                break

    return y_majority_pred


def get_treshold_voting(y_pred, y_pred_max_probs, mwe_dict, indices_test, class_tresholds, mwe_metadata, filepath):
    y_majority_pred = np.array([0 for _ in y_pred])

    for pred_ind, prediction in enumerate(y_pred):
        mwe_ind = indices_test[pred_ind]
        for ind_set in mwe_dict.values():
            if mwe_ind in ind_set:
                predictions_with_probs = [(y_pred[indices_test.tolist().index(label_ind)],
                                           y_pred_max_probs[indices_test.tolist().index(label_ind)]) for label_ind in
                                          ind_set if label_ind in indices_test]

                predictions = [elem[0] for elem in predictions_with_probs if elem[1] > class_tresholds[elem[0]]]

                # return class 0 if there aren't any predictions above treshold
                if len(predictions) == 0:
                    final_prediction = 0

                else:
                    final_prediction = int(s.mode(predictions)[0])

                y_majority_pred[pred_ind] = final_prediction

                write_prediction_result_to_file(filepath, mwe_ind, mwe_metadata, final_prediction)

                break

    return y_majority_pred


def main(args):
    if 'transformer_embeddings' in args:
        dataset_filepath = 'sentences_containing_mwe_from_kgr10_group_0_embeddings_1_layers_incomplete_mwe_in_sent.tsv'
        mwe_filepath = 'sentences_containing_mwe_from_kgr10_group_0_mwe_list_incomplete_mwe_in_sent.tsv'

        X_train, X_test, y_train, y_test, indices_train, indices_test, mwe_list, mwe_dict, mwe_metadata = load_transformer_embeddings_data(
            dataset_filepath, mwe_filepath)

    else:
        dataset_filepath = 'mwe_dataset.npy'
        # dataset_filepath = 'mwe_dataset_cbow.npy'
        # dataset_filepath = 'mwe_dataset_domain_balanced.npy'  # domain-balanced dataset

        X_train, X_test, y_train, y_test = load_data(dataset_filepath)

    # getting parts of train set by 5%
    for percentage in range(5, 101, 5):
        current_X_train = X_train[: int(len(X_train) * percentage * 0.01)]
        current_y_train = y_train[: int(len(y_train) * percentage * 0.01)]

        results_filepath = 'results_' + '_'.join(args) + str(percentage) + '.tsv'

        create_empty_file(results_filepath)

        if 'smote' in args:
            oversample = SMOTE()
            current_X_train, current_y_train = oversample.fit_resample(current_X_train, current_y_train)

        if 'pipeline' in args:
            over = SMOTE(sampling_strategy=0.1)
            under = RandomUnderSampler(sampling_strategy=0.5)
            steps = [('o', over), ('u', under)]
            pipeline = Pipeline(steps=steps)
            current_X_train, current_y_train = pipeline.fit_resample(current_X_train, current_y_train)

        if 'borderline_smote' in args:
            oversample = BorderlineSMOTE()
            current_X_train, current_y_train = oversample.fit_resample(current_X_train, current_y_train)

        if 'svm_smote' in args:
            oversample = SVMSMOTE()
            current_X_train, current_y_train = oversample.fit_resample(current_X_train, current_y_train)

        if 'adasyn' in args:
            oversample = ADASYN()
            current_X_train, current_y_train = oversample.fit_resample(current_X_train, current_y_train)

        if 'cnn' in args:
            print(f'current_X_train shape: {current_X_train.shape}')
            current_X_train = np.reshape(current_X_train, [current_X_train.shape[0], current_X_train.shape[1], 1])
            X_test = np.reshape(X_test, [X_test.shape[0], current_X_train.shape[1], 1])
            current_y_train = one_hot(current_y_train, depth=2)

            if 'eval' in args:
                eval_only = True
                model_path = args[2]
            else:
                eval_only = False
                model_path = None

            y_pred_probs = get_cnn_model_pred(current_X_train, current_y_train, X_test,
                                              eval_only=eval_only,
                                              model_path=model_path,
                                              input_shape=(current_X_train.shape[1], 1))

            y_pred_max_probs = [max(probs) for probs in y_pred_probs]

            y_pred = [np.argmax(probs) for probs in y_pred_probs]

        elif 'lr' in args:
            y_pred, y_pred_probs = get_lr_model_pred(current_X_train, current_y_train, X_test)

            y_pred_max_probs = [max(probs) for probs in y_pred_probs]

        elif 'rf' in args:
            y_pred, y_pred_probs = get_rf_model_pred(current_X_train, current_y_train, X_test)

            y_pred_max_probs = [max(probs) for probs in y_pred_probs]

        if 'majority_voting' in args:
            y_pred = get_majority_voting(y_pred, mwe_dict, indices_test, mwe_metadata, results_filepath)

        if 'weighted_voting' in args:
            y_pred = get_weighted_voting(y_pred, y_pred_max_probs, mwe_dict, indices_test, mwe_metadata, results_filepath)

        if 'treshold_voting' in args:
            for first_class_treshold in [0.1 * n for n in range(4, 10, 1)]:
                for second_class_treshold in [0.1 * n for n in range(4, 10, 1)]:
                    print(f'Evaluation results for tresholds:',
                          f'incorrect MWE treshold: {first_class_treshold}',
                          f'correct MWE treshold: {second_class_treshold}',
                          sep='\n')

                    class_tresholds = [first_class_treshold, second_class_treshold]

                    y_pred = get_treshold_voting(y_pred, y_pred_max_probs, mwe_dict, indices_test, class_tresholds, mwe_metadata, results_filepath)


                    write_results = 'transformer_embeddings' in args
                    get_evaluation_report(y_test, y_pred, write_results, indices_test, mwe_metadata, results_filepath)
        else:
            print(f'EVALUATION RESULTS FOR PERCENTAGE: {percentage}% OF TRAIN DATASET')

            write_results = 'transformer_embeddings' in args

            get_evaluation_report(y_test, y_pred, write_results, indices_test, mwe_metadata, results_filepath)


if __name__ == '__main__':
    main(sys.argv[1:])
