import argparse
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
from sklearn.utils import shuffle
from tensorflow import one_hot, config


def get_curr_time():
    return f'{datetime.datetime.now().strftime("%H:%M:%S")}'


def log_message(message):
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : {message}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--undersampled',
                        help='use undersampled version of the dataset',
                        action='store_true')

    args = parser.parse_args()

    return args


def limit_gpu_memory(memory_size):
    gpus = config.experimental.list_physical_devices('GPU')

    for device in gpus:
        config.experimental.set_memory_growth(device, True)

    if gpus:
        try:
            config.experimental.set_virtual_device_configuration(
                gpus[0], [
                    config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_size)
                ])
        except RuntimeError as e:
            print(e)


# not used in production
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


# not used in production
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


# not used in production
def create_empty_file(filepath):
    with open(filepath, 'w') as f:
        column_names_line = '\t'.join([
            'mwe', 'first_word_index', 'first_word_orth', 'first_word_lemma',
            'sentence', 'is_correct', 'model_prediction'
        ])
        f.write(f"{column_names_line}\n")
        pass


# not used in production
def write_line_to_file(filepath, line):
    with open(filepath, 'a') as f:
        f.write(f'{line}\n')


# move to utils
def get_evaluation_report(y_true, y_pred, y_pred_probs, full_df,
                          predictions_filepath, evaluation_filepath):
    test_df = full_df[full_df['dataset_type'] == 'test']

    columns_list = [
        column_name for column_name in list(test_df.columns)
        if 'emb' not in column_name
    ]

    y_pred_max_probs = [max(probs) for probs in y_pred_probs]

    report_df = test_df[columns_list]
    report_df['prediction'] = y_pred
    report_df['prediction_prob'] = y_pred_max_probs

    target_names = ['Incorrect MWE', 'Correct MWE']

    print(f'Saving prediction results to: {predictions_filepath}')

    report_df.to_csv(predictions_filepath, sep='\t', index=False)

    eval_report = classification_report(y_true,
                                        y_pred,
                                        target_names=target_names,
                                        output_dict=True,
                                        digits=4)

    eval_df = pd.DataFrame(eval_report).transpose().round(4)

    print(f'Saving evaluation results to: {evaluation_filepath}')

    eval_df.to_csv(f'{evaluation_filepath}', sep='\t')

    print(classification_report(y_true, y_pred, target_names=target_names))


# fix (it is in utils in file with 'scores_per_mwe') and move to utils
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


# fix and move to utils
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


# fix and move to utils
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


# divide into multiple experiment scripts
def main(args):
    gpu_memory_limit = 10240  # 4096 * 3
    # limit_gpu_memory(gpu_memory_limit)

    if 'bnc' in args:
        data_dir = os.path.join('storage', 'bnc', 'embeddings', 'transformer')
        storage_dir = os.path.join('storage', 'bnc')

        if 'pmi' in args and 'baseline' in args:
            print('pmi and baseline in args')
            train_filepath = os.path.join(
                data_dir,
                'correct_pmi_dataset_with_combined_embs_baseline_with_splits.tsv'
            )

            full_data_filepath = os.path.join(
                data_dir,
                'correct_pmi_dataset_with_combined_embs_baseline_with_splits.tsv'
            )

        if 'pmi' in args and 'diff_emb' in args:
            print('pmi and diff_emb in args')
            train_filepath = os.path.join(
                data_dir,
                'correct_pmi_dataset_with_combined_embs_diff_emb_with_splits.tsv'
            )

            full_data_filepath = os.path.join(
                data_dir,
                'correct_pmi_dataset_with_combined_embs_diff_emb_with_splits.tsv'
            )

        if 'pmi' in args and 'prod_emb' in args:
            print('pmi and prod_emb in args')
            train_filepath = os.path.join(
                data_dir,
                'correct_pmi_dataset_with_combined_embs_prod_emb_with_splits.tsv'
            )

            full_data_filepath = os.path.join(
                data_dir,
                'correct_pmi_dataset_with_combined_embs_prod_emb_with_splits.tsv'
            )

        if 'pmi' in args and 'mean_emb' in args:
            print('pmi and mean_emb in args')
            train_filepath = os.path.join(
                data_dir,
                'correct_pmi_dataset_with_combined_embs_mean_emb_with_splits.tsv'
            )

            full_data_filepath = os.path.join(
                data_dir,
                'correct_pmi_dataset_with_combined_embs_mean_emb_with_splits.tsv'
            )

        if 'dice' in args and 'baseline' in args:
            print('dice and baseline in args')
            train_filepath = os.path.join(
                data_dir,
                'correct_dice_dataset_with_combined_embs_baseline_with_splits.tsv'
            )

            full_data_filepath = os.path.join(
                data_dir,
                'correct_dice_dataset_with_combined_embs_baseline_with_splits.tsv'
            )

        if 'dice' in args and 'diff_emb' in args:
            print('dice and diff_emb in args')
            train_filepath = os.path.join(
                data_dir,
                'correct_dice_dataset_with_combined_embs_diff_emb_with_splits.tsv'
            )

            full_data_filepath = os.path.join(
                data_dir,
                'correct_dice_dataset_with_combined_embs_diff_emb_with_splits.tsv'
            )

        if 'dice' in args and 'prod_emb' in args:
            print('dice and prod_emb in args')
            train_filepath = os.path.join(
                data_dir,
                'correct_dice_dataset_with_combined_embs_prod_emb_with_splits.tsv'
            )

            full_data_filepath = os.path.join(
                data_dir,
                'correct_dice_dataset_with_combined_embs_prod_emb_with_splits.tsv'
            )

        if 'dice' in args and 'mean_emb' in args:
            print('dice and mean_emb in args')
            train_filepath = os.path.join(
                data_dir,
                'correct_dice_dataset_with_combined_embs_mean_emb_with_splits.tsv'
            )

            full_data_filepath = os.path.join(
                data_dir,
                'correct_dice_dataset_with_combined_embs_mean_emb_with_splits.tsv'
            )

        if 'chi2' in args and 'baseline' in args:
            print('chi2 and baseline in args')
            train_filepath = os.path.join(
                data_dir,
                'correct_chi2_dataset_with_combined_embs_baseline_with_splits.tsv'
            )

            full_data_filepath = os.path.join(
                data_dir,
                'correct_chi2_dataset_with_combined_embs_baseline_with_splits.tsv'
            )

        if 'chi2' in args and 'diff_emb' in args:
            print('chi2 and diff_emb in args')
            train_filepath = os.path.join(
                data_dir,
                'correct_chi2_dataset_with_combined_embs_diff_emb_with_splits.tsv'
            )

            full_data_filepath = os.path.join(
                data_dir,
                'correct_chi2_dataset_with_combined_embs_diff_emb_with_splits.tsv'
            )

        if 'chi2' in args and 'prod_emb' in args:
            print('chi2 and prod_emb in args')
            train_filepath = os.path.join(
                data_dir,
                'correct_chi2_dataset_with_combined_embs_prod_emb_with_splits.tsv'
            )

            full_data_filepath = os.path.join(
                data_dir,
                'correct_chi2_dataset_with_combined_embs_prod_emb_with_splits.tsv'
            )

        if 'chi2' in args and 'mean_emb' in args:
            print('chi2 and mean_emb in args')
            train_filepath = os.path.join(
                data_dir,
                'correct_chi2_dataset_with_combined_embs_mean_emb_with_splits.tsv'
            )

            full_data_filepath = os.path.join(
                data_dir,
                'correct_chi2_dataset_with_combined_embs_mean_emb_with_splits.tsv'
            )

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

    if 'kgr10_new_incorrect' in args:
        storage_dir = os.path.join('storage', 'kgr10_containing_mewex_results')
        data_dir = os.path.join('storage', 'kgr10_containing_mewex_results',
                                'embeddings', 'transformer')

        train_filepath = os.path.join(
            data_dir,
            'sentences_containing_mwe_from_kgr10_group_0_embeddings_1_layers_incomplete_mwe_in_sent_with_splits_undersampled_ratio_1_3_with_new_incorrect_mwe_with_splits_abs_prod_emb.tsv'
        )

        full_data_filepath = os.path.join(
            data_dir,
            'sentences_containing_mwe_from_kgr10_group_0_embeddings_1_layers_incomplete_mwe_in_sent_with_splits_undersampled_ratio_1_3_with_new_incorrect_mwe_with_splits_abs_prod_emb.tsv'
        )

    if 'undersampled' in args:
        train_filepath = f'{train_filepath.split(".")[0]}_undersampled_ratio_1.3_fixed_nDD.tsv'

        full_data_filepath = f'{full_data_filepath.split(".")[0]}_undersampled_ratio_1.3_fixed_nDD.tsv'

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
        pred_results_dir,
        f'{datetime.datetime.now().date()}_{datetime.datetime.now().strftime("%H:%M:%S")}_prediction_results_'
        + '_'.join(args) + '.tsv')

    eval_results_dir = os.path.join(*data_dir.split('/')[:-2],
                                    'evaluation_results')

    if not os.path.exists(eval_results_dir):
        os.mkdir(eval_results_dir)

    eval_results_filepath = os.path.join(
        eval_results_dir,
        f'{datetime.datetime.now().date()}_{datetime.datetime.now().strftime("%H:%M:%S")}_evaluation_results_'
        + '_'.join(args) + '.tsv')

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
            # 'sentences_containing_mwe_from_kgr10_group_0_embeddings_1_layers_incomplete_mwe_in_sent_with_splits_svm.tsv'
            'parseme_pl_embeddings_train_svm.tsv')

    if 'adasyn' in args:
        train_filepath = os.path.join(
            data_dir, 'parseme_pl_embeddings_train_adasyn.tsv')

    print(f'{get_curr_time()} : Loading data...')
    train_df = pd.read_csv(train_filepath,
                           sep='\t',
                           on_bad_lines='skip',
                           nrows=None)

    full_df = pd.read_csv(full_data_filepath,
                          sep='\t',
                          on_bad_lines='skip',
                          nrows=None)

    if 'bnc' in args:
        train_df = train_df.dropna()
        full_df = full_df.dropna()

        emb_column = [
            column_name for column_name in list(train_df.columns)
            if column_name in ['baseline', 'diff_emb', 'prod_emb', 'mean_emb']
        ][0]
        train_df = train_df.rename(columns={emb_column: 'combined_embedding'})

        full_df = full_df.rename(columns={emb_column: 'combined_embedding'})

        train_df = train_df[~train_df['combined_embedding'].str.contains("nan"
                                                                         )]

        full_df = full_df[~full_df['combined_embedding'].str.contains("nan")]

    print(f'{get_curr_time()} : Getting train data...')
    if ('smote' in args or 'borderline_smote' in args or 'svm_smote' in args
            or 'adasyn' in args):
        X_train = train_df['combined_embedding'].tolist()

        y_train = train_df['is_correct'].tolist()

    elif 'kgr10_new_incorrect' in args:
        X_train = train_df[train_df['dataset_type'] ==
                           'train']['abs_prod_emb'].tolist()

        y_train = train_df[train_df['dataset_type'] ==
                           'train']['is_correct'].tolist()

    else:
        X_train = train_df[train_df['dataset_type'] ==
                           'train']['combined_embedding'].tolist()

        y_train = train_df[train_df['dataset_type'] ==
                           'train']['is_correct'].tolist()

    if ('smote' in args or 'borderline_smote' in args or 'svm_smote' in args
            or 'adasyn' in args):
        X_train = np.array([
            np.array([float(elem) for elem in embedding.split(',')])
            for embedding in X_train
        ])

    elif 'bnc' not in args and 'kgr10_new_incorrect' not in args:
        print(f'{get_curr_time()} : Ommiting difference vectors...')
        X_train = np.array([
            np.array([
                float(elem) for elem in (embedding.split(',')[:768 * 2] +
                                         embedding.split(',')[768 * 3:])
            ]) for embedding in X_train
        ])

    else:
        X_train = np.array([
            np.array([float(elem) for elem in embedding.split(',')])
            for embedding in X_train
        ])

    y_train = np.array([int(elem) for elem in y_train])
    print(f'{get_curr_time()} : Getting dev data...')

    X_dev = full_df[full_df['dataset_type'] ==
                    'dev']['combined_embedding'].tolist()

    if 'kgr10_new_incorrect' in args:
        X_dev = full_df[full_df['dataset_type'] ==
                        'dev']['abs_prod_emb'].tolist()

    if 'bnc' not in args or 'kgr10_new_incorrect' not in args:
        X_dev = np.array([
            np.array([
                float(elem) for elem in (embedding.split(',')[:768 * 2] +
                                         embedding.split(',')[768 * 3:])
            ]) for embedding in X_dev
        ])
    else:
        X_dev = np.array([
            np.array([float(elem) for elem in embedding.split(',')])
            for embedding in X_dev
        ])

    y_dev = full_df[full_df['dataset_type'] == 'dev']['is_correct'].tolist()

    y_dev = np.array([int(elem) for elem in y_dev])
    print(f'{get_curr_time()} : Getting test data...')
    X_test = full_df[full_df['dataset_type'] ==
                     'test']['combined_embedding'].tolist()

    if 'kgr10_new_incorrect' in args:
        X_test = full_df[full_df['dataset_type'] ==
                         'test']['abs_prod_emb'].tolist()

    if 'bnc' not in args or 'kgr10_new_incorrect' not in args:
        X_test = np.array([
            np.array([
                float(elem) for elem in (embedding.split(',')[:768 * 2] +
                                         embedding.split(',')[768 * 3:])
            ]) for embedding in X_test
        ])

    else:
        X_test = np.array([
            np.array([float(elem) for elem in embedding.split(',')])
            for embedding in X_test
        ])

    y_test = full_df[full_df['dataset_type'] == 'test']['is_correct'].tolist()

    y_test = np.array([int(elem) for elem in y_test])

    # X_train, y_train = shuffle(X_train, y_train)
    # X_dev, y_dev = shuffle(X_dev, y_dev)
    # X_test, y_test = shuffle(X_test, y_test)

    # X_train_bad_embs = np.array(
    #     [i for i, emb in enumerate(X_train) if np.all((emb == 0.0))])

    # X_train = np.array(
    #     [emb for i, emb in enumerate(X_train) if i not in X_train_bad_embs])
    # y_train = np.array([
    #     label for i, label in enumerate(y_train) if i not in X_train_bad_embs
    # ])

    # X_dev_bad_embs = np.array(
    #     [i for i, emb in enumerate(X_dev) if np.all((emb == 0.0))])

    # X_dev = np.array(
    #     [emb for i, emb in enumerate(X_dev) if i not in X_dev_bad_embs])
    # y_dev = np.array(
    #     [label for i, label in enumerate(y_dev) if i not in X_dev_bad_embs])

    # X_test_bad_embs = np.array(
    #     [i for i, emb in enumerate(X_test) if np.all((emb == 0.0))])

    # X_test = np.array(
    #     [emb for i, emb in enumerate(X_test) if i not in X_test_bad_embs])
    # y_test = np.array(
    #     [label for i, label in enumerate(y_test) if i not in X_test_bad_embs])

    print(f'X_train shape: {X_train.shape}',
          f'y_train length: {len(y_train)}',
          f'X_dev shape: {X_dev.shape}',
          f'y_dev length: {len(y_dev)}',
          f'X_test shape: {X_test.shape}',
          f'y_test length: {len(y_test)}',
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
                                          input_shape=(X_train.shape[1], 1),
                                          num_epochs=20)

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
                get_evaluation_report(y_test, y_pred, y_pred_probs, full_df,
                                      pred_results_filepath,
                                      eval_results_filepath)

    else:
        print(
            f'{get_curr_time()} : Generating and saving evaluation results...')
        get_evaluation_report(y_test, y_pred, y_pred_probs, full_df,
                              pred_results_filepath, eval_results_filepath)


if __name__ == '__main__':
    main(sys.argv[1:])
