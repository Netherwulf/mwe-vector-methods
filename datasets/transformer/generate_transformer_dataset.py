import csv
import os
import pickle as pkl
import re
import sys

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
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


# older version for KGR10 (Słowosieć) dataset
# def get_mwe(mwe_file, idx_list):
#     with open(mwe_file, 'r', errors='replace') as in_file:
#         content = in_file.readlines()
#
#         mwe_list = np.array([line.strip().split('\t')[0] for ind, line in enumerate(content) if ind in idx_list])
#
#         mwe_metadata = np.array([line.strip().split('\t') for ind, line in enumerate(content) if ind in idx_list])
#
#         mwe_dict = {}
#
#         for mwe_ind, mwe in enumerate(mwe_list):
#             if mwe not in mwe_dict.keys():
#                 mwe_dict[mwe] = np.array([mwe_ind])
#
#             else:
#                 mwe_dict[mwe] = np.append(mwe_dict[mwe], mwe_ind)
#
#         return mwe_list, mwe_dict, mwe_metadata


def get_mwe(mwe_file):
    df = pd.read_csv(mwe_file, sep='\t')

    mwe_list = np.array([mwe for mwe in df['mwe'].tolist()])

    mwe_metadata = np.array(
        [[
            mwe_type, first_word, first_word_id, second_word, second_word_id,
            mwe, sentence, is_correct, complete_mwe_in_sent
        ] for mwe_type, first_word, first_word_id, second_word, second_word_id,
         mwe, sentence, is_correct, complete_mwe_in_sent in zip(
             df['mwe_type'].tolist(), df['first_word'].tolist(),
             df['first_word_id'].tolist(), df['second_word'].tolist(),
             df['second_word_id'].tolist(), df['mwe'].tolist(),
             df['sentence'].tolist(), df['is_correct'].tolist(),
             df['complete_mwe_in_sent'].tolist())])

    mwe_dict = {}

    for mwe_ind, mwe in enumerate(mwe_list):
        if mwe not in mwe_dict.keys():
            mwe_dict[mwe] = np.array([mwe_ind])

        else:
            mwe_dict[mwe] = np.append(mwe_dict[mwe], mwe_ind)

    return mwe_list, mwe_dict, mwe_metadata


# older version for KGR10 (Słowosieć) dataset
# def load_transformer_embeddings_data(dataset_file, mwe_file):
#     print(f'Reading file: {dataset_file.split("/")[-1]}')
#     df = pd.read_csv(dataset_file, sep='\t', header=None)
#
#     print('Generating embeddings list...')
#     df[4] = df[0] + ',' + df[1] + ',' + df[2]
#
#     embeddings_list = [elem.split(',') for elem in df[4].tolist()]
#
#     correct_idx_list = np.array([ind for ind, sentence in enumerate(embeddings_list) if 'tensor(nan)' not in sentence])
#
#     embeddings_list = [([float(re.findall(r"[-+]?\d*\.\d+|\d+", val)[0]) for val in sentence], label) for
#                        sentence, label in zip(embeddings_list, df[3].tolist()) if 'tensor(nan)' not in sentence]
#
#     mwe_list, mwe_dict, mwe_metadata = get_mwe(mwe_file, correct_idx_list)
#
#     X = np.array([elem[0] for elem in embeddings_list])
#
#     y = np.array([elem[1] for elem in embeddings_list])
#     y = y.astype(int)
#
#     indices = np.arange(X.shape[0])
#
#     X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X,
#                                                                                      y,
#                                                                                      indices,
#                                                                                      test_size=0.20,
#                                                                                      random_state=42)
#
#     return X_train, X_test, y_train, y_test, indices_train, indices_test, mwe_list, mwe_dict, mwe_metadata


def load_transformer_embeddings_data(dataset_file):
    print(f'Reading file: {dataset_file.split("/")[-1]}')
    df = pd.read_csv(dataset_file, sep='\t')

    print('Generating embeddings list...')
    mwe_embedding = np.array([
        np.array([float(elem) for elem in line.split(',')])
        for line in df['mwe_embedding'].tolist()
    ])
    first_word_only_embedding = np.array([
        np.array([float(elem) for elem in line.split(',')])
        for line in df['first_word_only_embedding'].tolist()
    ])
    second_word_only_embedding = np.array([
        np.array([float(elem) for elem in line.split(',')])
        for line in df['second_word_only_embedding'].tolist()
    ])
    first_word_mwe_emb_diff = np.array([
        np.array([float(elem) for elem in line.split(',')])
        for line in df['first_word_mwe_emb_diff'].tolist()
    ])
    second_word_mwe_emb_diff = np.array([
        np.array([float(elem) for elem in line.split(',')])
        for line in df['second_word_mwe_emb_diff'].tolist()
    ])
    first_word_mwe_emb_abs_diff = np.array([
        np.array([float(elem) for elem in line.split(',')])
        for line in df['first_word_mwe_emb_abs_diff'].tolist()
    ])
    second_word_mwe_emb_abs_diff = np.array([
        np.array([float(elem) for elem in line.split(',')])
        for line in df['second_word_mwe_emb_abs_diff'].tolist()
    ])
    first_word_mwe_emb_prod = np.array([
        np.array([float(elem) for elem in line.split(',')])
        for line in df['first_word_mwe_emb_prod'].tolist()
    ])
    second_word_mwe_emb_prod = np.array([
        np.array([float(elem) for elem in line.split(',')])
        for line in df['second_word_mwe_emb_prod'].tolist()
    ])

    embeddings_list = np.hstack(
        (mwe_embedding, first_word_only_embedding, second_word_only_embedding,
         first_word_mwe_emb_diff, second_word_mwe_emb_diff,
         first_word_mwe_emb_abs_diff, second_word_mwe_emb_abs_diff,
         first_word_mwe_emb_prod, second_word_mwe_emb_prod))

    embeddings_list = np.array([
        ','.join([str(elem) for elem in embedding])
        for embedding in embeddings_list
    ])

    df['combined_embedding'] = embeddings_list

    # embeddings_list = [
    #     (embedding, label)
    #     for embedding, label in zip(embeddings_list, df['is_correct'].tolist())
    # ]

    # mwe_list, mwe_dict, mwe_metadata = get_mwe(dataset_file)

    # X = np.array([elem[0] for elem in embeddings_list])

    # y = np.array([elem[1] for elem in embeddings_list])
    # y = y.astype(int)

    # indices = np.arange(X.shape[0])

    # X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    #     X, y, indices, test_size=0.20, random_state=42)

    # return X_train, X_test, y_train, y_test, indices_train, indices_test, mwe_list, mwe_dict, mwe_metadata
    return df


def create_empty_file(filepath):
    with open(filepath, 'w') as f:
        column_names_line = '\t'.join([
            'mwe_type', 'first_word', 'first_word_id', 'second_word',
            'second_word_id', 'mwe', 'sentence', 'is_correct',
            'complete_mwe_in_sent'
        ])
        f.write(f"{column_names_line}\n")
        pass


def write_line_to_file(filepath, line):
    with open(filepath, 'a') as f:
        f.write(f'{line}\n')


def save_list_of_lists(filepath, list_of_lists):
    with open(filepath, "w") as f:
        wr = csv.writer(f)
        wr.writerows(list_of_lists)


def save_list(filepath, list_to_save):
    list_string = ','.join([str(elem) for elem in list_to_save])

    with open(filepath, "w") as f:
        f.write(f'{list_string}\n')


def save_dict(filepath, dict_to_save):
    with open(filepath, 'wb') as f:
        pkl.dump(dict_to_save, f)


def load_dict(filepath):
    with open(filepath, 'rb') as f:
        loaded_dict = pkl.load(f)
    return loaded_dict


def get_smote_oversampler(smote_key):
    smote_dict = {
        'smote': SMOTE(),
        'borderline': BorderlineSMOTE(),
        'svm': SVMSMOTE(),
        'adasyn': ADASYN()
    }

    return smote_dict[smote_key]


def main(args):
    # dataset_filepath = 'sentences_containing_mwe_from_kgr10_group_0_embeddings_1_layers_incomplete_mwe_in_sent.tsv'
    # mwe_filepath = 'sentences_containing_mwe_from_kgr10_group_0_mwe_list_incomplete_mwe_in_sent.tsv'
    # dataset_filepath = 'parseme_merged_mwes_embeddings_1_layers_complete_mwe_in_sent.tsv'

    # result_dir_name = 'parseme_transformer_embeddings_pl'
    result_dir_name = os.path.join('..', '..', 'storage', 'parseme', 'pl',
                                   'embeddings', 'transformer')

    dataset_filepath = os.path.join(
        result_dir_name,
        'parseme_data_embeddings_1_layers_complete_mwe_in_sent.tsv')

    # X_train, X_test, y_train, y_test, indices_train, indices_test, mwe_list, mwe_dict, mwe_metadata = load_transformer_embeddings_data(
    #     dataset_filepath)
    df = load_transformer_embeddings_data(dataset_filepath)

    if not os.path.exists(result_dir_name):
        os.mkdir(result_dir_name)

    # print('Saving train data...')
    # save_list_of_lists(os.path.join(result_dir_name, "X_train.csv"), X_train)
    # save_list(os.path.join(result_dir_name, "y_train.csv"), y_train)

    df.to_csv(os.path.join(result_dir_name, 'parseme_pl_embeddings.tsv'),
              sep='\t',
              index=False)

    # SMOTE, Borderline SMOTE, SVM-SMOTE and ADASYN dataset variants
    for smote_type in ['smote', 'borderline', 'svm', 'adasyn']:
        print(f'Generating {smote_type} dataset variant...')
        oversample = get_smote_oversampler(smote_type)

        X_train = df[df['dataset_type'] ==
                     'train']['combined_embedding'].tolist()

        X_train = np.array([
            np.array([float(elem) for elem in embedding.split(',')])
            for embedding in X_train
        ])

        y_train = df[df['dataset_type'] == 'train']['is_correct'].tolist()

        transformed_X_train, transformed_y_train = oversample.fit_resample(
            X_train, y_train)

        transformed_X_train = np.array([
            ','.join([str(elem) for elem in embedding])
            for embedding in transformed_X_train
        ])

        transformed_y_train = np.array(
            [str(label) for label in transformed_y_train])

        transformed_df = pd.DataFrame({
            'combined_embedding': transformed_X_train,
            'is_correct': transformed_y_train
        })

        transformed_df.to_csv(os.path.join(
            result_dir_name, f'parseme_pl_embeddings_train_{smote_type}.tsv'),
                              sep='\t',
                              index=False)

        # save_list_of_lists(
        #     os.path.join(result_dir_name, f"X_train_{smote_type}.csv"),
        #     transformed_X_train)
        # save_list(os.path.join(result_dir_name, f"y_train_{smote_type}.csv"),
        #           transformed_y_train_smote)

    # Imblearn Pipeline dataset variant
    # print('Generating Imblearn Pipeline dataset variant...')
    # over = SMOTE(sampling_strategy=0.1)
    # under = RandomUnderSampler(sampling_strategy=0.5)
    # steps = [('o', over), ('u', under)]
    # pipeline = Pipeline(steps=steps)
    # X_train_pipeline, y_train_pipeline = pipeline.fit_resample(X_train, y_train)
    # save_list_of_lists(os.path.join(result_dir_name, "X_train_pipeline.csv"), X_train_pipeline)
    # save_list(os.path.join(result_dir_name, "y_train_pipeline.csv"), y_train_pipeline)

    # print('Saving test data...')
    # save_list_of_lists(os.path.join(result_dir_name, "X_test.csv"), X_test)
    # save_list(os.path.join(result_dir_name, "y_test.csv"), y_test)

    # print('Saving indices files...')
    # save_list(os.path.join(result_dir_name, "indices_train.csv"),
    #           indices_train)
    # save_list(os.path.join(result_dir_name, "indices_test.csv"), indices_test)

    # print('Saving mwe dict...')
    # save_dict(os.path.join(result_dir_name, 'mwe_dict.pkl'), mwe_dict)

    # print('Saving mwe list...')
    # save_list(os.path.join(result_dir_name, 'mwe_list.csv'), mwe_list)

    # print('Saving mwe metadata...')
    # save_list_of_lists(os.path.join(result_dir_name, 'mwe_metadata.csv'),
    #                    mwe_metadata)


if __name__ == '__main__':
    main(sys.argv[1:])
