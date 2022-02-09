import csv
import datetime
import os
import pickle as pkl
import re
import statistics
import sys

import numpy as np
import pandas as pd

from models.cnn import get_cnn_model_pred
from models.logistic_regression import get_lr_model_pred
from models.random_forest import get_rf_model_pred

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from scipy import stats as s
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow import one_hot


def get_curr_time():
    return f'{datetime.datetime.now().strftime("%H:%M:%S")}'


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

        mwe_list = np.array([
            line.strip().split('\t')[0] for ind, line in enumerate(content)
            if ind in idx_list
        ])

        mwe_metadata = np.array([
            line.strip().split('\t') for ind, line in enumerate(content)
            if ind in idx_list
        ])

        mwe_dict = {}

        for mwe_ind, mwe in enumerate(mwe_list):
            if mwe not in mwe_dict.keys():
                mwe_dict[mwe] = np.array([mwe_ind])

            else:
                mwe_dict[mwe] = np.append(mwe_dict[mwe], mwe_ind)

        return mwe_list, mwe_dict, mwe_metadata


def create_empty_file(filepath):
    with open(filepath, 'w') as f:
        column_names_line = '\t'.join([
            'mwe', 'first_word_index', 'first_word_orth', 'first_word_lemma',
            'sentence', 'is_correct', 'model_prediction'
        ])
        f.write(f"{column_names_line}\n")
        pass


def write_line_to_file(filepath, line):
    with open(filepath, 'a') as f:
        f.write(f'{line}\n')


def get_evaluation_report(y_true, y_pred, full_df, predictions_filepath,
                          evaluation_filepath):
    test_df = full_df[full_df['dataset_type'] == 'test']

    columns_list = [
        column_name for column_name in list(test_df.columns)
        if 'emb' not in column_name
    ]

    report_df = test_df[columns_list]
    report_df['prediction'] = y_pred

    target_names = ['Incorrect MWE', 'Correct MWE']

    print(f'Saving prediction results to: {predictions_filepath}')

    report_df.to_csv(predictions_filepath, sep='\t', index=False)

    eval_report = classification_report(y_true,
                                        y_pred,
                                        target_names=target_names,
                                        output_dict=True,
                                        digits=4)

    eval_df = pd.DataFrame(eval_report).transpose()

    print(f'Saving evaluation results to: {evaluation_filepath}')

    eval_df.to_csv(f'{evaluation_filepath}', sep='\t')

    print(classification_report(y_true, y_pred, target_names=target_names))


def get_majority_voting(y_pred, full_df, filepath):
    y_majority_pred = np.array([0 for _ in y_pred])

    df_test = full_df[full_df['dataset_type'] == 'test'].reset_index(drop=True)

    mwe_list = df_test['mwe'].tolist()

    for pred_ind, prediction in enumerate(y_pred):
        mwe = mwe_list[pred_ind]
        ind_set = df_test.index[df_test['mwe'] == mwe]

        predictions = [y_pred[mwe_ind] for mwe_ind in ind_set]

        final_prediction = int(s.mode(predictions)[0])

        y_majority_pred[pred_ind] = final_prediction

        break

    return y_majority_pred


def get_weighted_voting(y_pred, y_pred_max_probs, full_df, filepath):
    y_majority_pred = np.array([0 for _ in y_pred])

    df_test = full_df[full_df['dataset_type'] == 'test'].reset_index(drop=True)

    mwe_list = df_test['mwe'].tolist()

    for pred_ind, prediction in enumerate(y_pred):
        mwe = mwe_list[pred_ind]
        ind_set = df_test.index[df_test['mwe'] == mwe]

        predictions_with_probs = [(y_pred[label_ind],
                                   y_pred_max_probs[label_ind])
                                  for label_ind in ind_set]

        weights_per_class = [0.0 for _ in range(2)]

        for class_id in range(len(weights_per_class)):
            weights_per_class[class_id] = sum([
                elem[1] for elem in predictions_with_probs
                if elem[0] == class_id
            ])

        final_prediction = int(np.argmax(weights_per_class))

        y_majority_pred[pred_ind] = final_prediction

        break

    return y_majority_pred


def get_treshold_voting(y_pred, y_pred_max_probs, full_df, class_tresholds,
                        filepath):
    y_majority_pred = np.array([0 for _ in y_pred])

    for pred_ind, prediction in enumerate(y_pred):
        mwe_ind = indices_test[pred_ind]
        for ind_set in mwe_dict.values():
            if mwe_ind in ind_set:
                predictions_with_probs = [
                    (y_pred[indices_test.tolist().index(label_ind)],
                     y_pred_max_probs[indices_test.tolist().index(label_ind)])
                    for label_ind in ind_set if label_ind in indices_test
                ]

                predictions = [
                    elem[0] for elem in predictions_with_probs
                    if elem[1] > class_tresholds[elem[0]]
                ]

                # return class 0 if there aren't any predictions above treshold
                if len(predictions) == 0:
                    final_prediction = 0

                else:
                    final_prediction = int(s.mode(predictions)[0])

                y_majority_pred[pred_ind] = final_prediction

                write_prediction_result_to_file(filepath, mwe_ind,
                                                mwe_metadata, final_prediction)

                break

    return y_majority_pred


def main(args):

    if 'parseme' in args and 'transformer_embeddings' in args:
        data_dir = os.path.join('storage', 'parseme', 'pl', 'embeddings',
                                'transformer')

    if 'parseme' in args and 'fasttext_embeddings' in args:
        data_dir = os.path.join('storage', 'parseme', 'pl', 'embeddings',
                                'fasttext_dupplicates')

    if 'kgr10' in args:
        storage_dir = os.path.join('storage', 'kgr10')

    if 'kgr10' in args and 'transformer_embeddings' in args:
        data_dir = os.path.join('storage', 'kgr10', 'embeddings',
                                'transformer')

        train_filepath = os.path.join(
            data_dir,
            'sentences_containing_mwe_from_kgr10_group_0_embeddings_1_layers_incomplete_mwe_in_sent_with_splits.tsv'
        )

        full_data_filepath = os.path.join(
            data_dir,
            'sentences_containing_mwe_from_kgr10_group_0_embeddings_1_layers_incomplete_mwe_in_sent_with_splits.tsv'
        )

    if 'parseme' in args:
        storage_dir = os.path.join('storage', 'parseme', 'pl')
        train_filepath = os.path.join(data_dir, 'parseme_pl_embeddings.tsv')

        full_data_filepath = os.path.join(data_dir,
                                          'parseme_pl_embeddings.tsv')

    pred_results_dir = os.path.join(*data_dir.split('/')[:-2],
                                    'prediction_results')

    if not os.path.exists(pred_results_dir):
        os.mkdir(pred_results_dir)

    pred_results_filepath = os.path.join(
        pred_results_dir, 'prediction_results_' + '_'.join(args) + '.tsv')

    eval_results_dir = os.path.join(*data_dir.split('/')[:-2],
                                    'evaluation_results')

    if not os.path.exists(eval_results_dir):
        os.mkdir(eval_results_dir)

    eval_results_filepath = os.path.join(
        eval_results_dir, 'evaluation_results_' + '_'.join(args) + '.tsv')

    if 'smote' in args:
        train_filepath = os.path.join(
            data_dir,
            'sentences_containing_mwe_from_kgr10_group_0_embeddings_1_layers_incomplete_mwe_in_sent_with_splits_smote.tsv'
        )

    if 'borderline_smote' in args:
        train_filepath = os.path.join(
            data_dir, 'parseme_pl_embeddings_train_borderline.tsv')

    if 'svm_smote' in args:
        train_filepath = os.path.join(
            data_dir,
            'sentences_containing_mwe_from_kgr10_group_0_embeddings_1_layers_incomplete_mwe_in_sent_with_splits_svm.tsv'
        )

    if 'adasyn' in args:
        train_filepath = os.path.join(
            data_dir, 'parseme_pl_embeddings_train_adasyn.tsv')

    print(f'{get_curr_time()} : Loading data...')
    train_df = pd.read_csv(train_filepath, sep='\t', on_bad_lines='skip')
    #    nrows=100)
    full_df = pd.read_csv(full_data_filepath, sep='\t', on_bad_lines='skip')
    #   nrows=100)
    print(f'{get_curr_time()} : Getting train data...')
    if ('smote' in args or 'borderline_smote' in args or 'svm_smote' in args
            or 'adasyn' in args):
        X_train = train_df['combined_embedding'].tolist()

        y_train = train_df['is_correct'].tolist()

    else:
        num_train_samples = 200000

        X_train = train_df[train_df['dataset_type'] == 'train'][
            'combined_embedding'].tolist()[:num_train_samples]

        y_train = train_df[train_df['dataset_type'] ==
                           'train']['is_correct'].tolist()[:num_train_samples]

    if ('smote' in args or 'borderline_smote' in args or 'svm_smote' in args
            or 'adasyn' in args):
        X_train = np.array([
            np.array([float(elem) for elem in embedding.split(',')])
            for embedding in X_train
        ])

    else:
        print(f'{get_curr_time()} : Ommiting difference vectors...')
        X_train = np.array([
            np.array([
                float(elem) for elem in (embedding.split(',')[:768 * 2] +
                                         embedding.split(',')[768 * 3:])
            ]) for embedding in X_train
        ])

    y_train = np.array([int(elem) for elem in y_train])
    print(f'{get_curr_time()} : Getting dev data...')
    X_dev = full_df[full_df['dataset_type'] ==
                    'dev']['combined_embedding'].tolist()

    X_dev = np.array([
        np.array([
            float(elem) for elem in (embedding.split(',')[:768 * 2] +
                                     embedding.split(',')[768 * 3:])
        ]) for embedding in X_dev
    ])

    y_dev = full_df[full_df['dataset_type'] == 'dev']['is_correct'].tolist()

    y_dev = np.array([int(elem) for elem in y_dev])
    print(f'{get_curr_time()} : Getting test data...')
    X_test = full_df[full_df['dataset_type'] ==
                     'test']['combined_embedding'].tolist()

    X_test = np.array([
        np.array([
            float(elem) for elem in (embedding.split(',')[:768 * 2] +
                                     embedding.split(',')[768 * 3:])
        ]) for embedding in X_test
    ])

    y_test = full_df[full_df['dataset_type'] == 'test']['is_correct'].tolist()

    y_test = np.array([int(elem) for elem in y_test])

    print(f'X_train shape: {X_train.shape}',
          f'X_dev shape: {X_dev.shape}',
          f'X_test shape: {X_test.shape}',
          sep='\n')

    if 'undersampling' in args:
        undersample = RandomUnderSampler(sampling_strategy='majority')
        X_train, y_train = undersample.fit_resample(X_train, y_train)

    if 'diff_vector_only' in args:
        X_train = np.array([embedding[768 * 2:] for embedding in X_train])
        X_test = np.array([embedding[768 * 2:] for embedding in X_test])

    if 'cnn' in args:
        print(f'{get_curr_time()} : Converting data for CNN model...')
        X_train = np.array(X_train)
        X_dev = np.array(X_dev)
        X_test = np.array(X_test)
        X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], 1])
        X_dev = np.reshape(X_dev, [X_dev.shape[0], X_dev.shape[1], 1])
        X_test = np.reshape(X_test, [X_test.shape[0], X_test.shape[1], 1])
        y_train = one_hot(y_train, depth=2)
        y_dev = one_hot(y_dev, depth=2)

        if 'eval' in args:
            eval_only = True
            model_path = args[2]
        else:
            eval_only = False
            model_path = None
        print(f'{get_curr_time()} : Training CNN model...')
        y_pred_probs = get_cnn_model_pred(X_train,
                                          y_train,
                                          X_dev,
                                          y_dev,
                                          X_test,
                                          storage_dir,
                                          eval_only=eval_only,
                                          model_path=model_path,
                                          input_shape=(X_train.shape[1], 1))

        y_pred_max_probs = [max(probs) for probs in y_pred_probs]
        print(f'{get_curr_time()} : Getting predictions on test set...')
        y_pred = [np.argmax(probs) for probs in y_pred_probs]

    elif 'lr' in args:
        y_pred, y_pred_probs = get_lr_model_pred(X_train, y_train, X_test)

        y_pred_max_probs = [max(probs) for probs in y_pred_probs]

    elif 'rf' in args:
        y_pred, y_pred_probs = get_rf_model_pred(X_train, y_train, X_test)

        y_pred_max_probs = [max(probs) for probs in y_pred_probs]

    if 'majority_voting' in args:
        print(f'{get_curr_time()} : Generating majority voting results...')
        y_pred = get_majority_voting(y_pred, full_df, pred_results_filepath)

    if 'weighted_voting' in args:
        print(f'{get_curr_time()} : Generating weighted voting results...')
        y_pred = get_weighted_voting(y_pred, y_pred_max_probs, full_df,
                                     pred_results_filepath)

    if 'treshold_voting' in args:
        for first_class_treshold in [0.1 * n for n in range(4, 10, 1)]:
            for second_class_treshold in [0.1 * n for n in range(4, 10, 1)]:
                print('Evaluation results for tresholds:',
                      f'incorrect MWE treshold: {first_class_treshold}',
                      f'correct MWE treshold: {second_class_treshold}',
                      sep='\n')

                class_tresholds = [first_class_treshold, second_class_treshold]

                y_pred = get_treshold_voting(y_pred, y_pred_max_probs, full_df,
                                             class_tresholds,
                                             pred_results_filepath)
                print(
                    f'{get_curr_time()} : Generating and saving evaluation results...'
                )
                get_evaluation_report(y_test, y_pred, full_df,
                                      pred_results_filepath,
                                      eval_results_filepath)

    else:
        print(
            f'{get_curr_time()} : Generating and saving evaluation results...')
        get_evaluation_report(y_test, y_pred, full_df, pred_results_filepath,
                              eval_results_filepath)


if __name__ == '__main__':
    main(sys.argv[1:])
