import csv
import os
import pickle as pkl
import re
import sys

import numpy as np
import pandas as pd

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


def main(args):
    dataset_filepath = 'sentences_containing_mwe_from_kgr10_group_0_embeddings_1_layers_incomplete_mwe_in_sent.tsv'
    mwe_filepath = 'sentences_containing_mwe_from_kgr10_group_0_mwe_list_incomplete_mwe_in_sent.tsv'

    X_train, X_test, y_train, y_test, indices_train, indices_test, mwe_list, mwe_dict, mwe_metadata = load_transformer_embeddings_data(
        dataset_filepath, mwe_filepath)

    result_dir_name = 'transformer_embeddings_dataset'

    os.mkdir(result_dir_name)

    save_list_of_lists(os.path.join(result_dir_name, "X_train.csv"), X_train)
    save_list(os.path.join(result_dir_name, "y_train.csv"), y_train)

    save_list_of_lists(os.path.join(result_dir_name, "X_test.csv"), X_test)
    save_list(os.path.join(result_dir_name, "y_test.csv"), y_test)

    save_list(os.path.join(result_dir_name, "indices_train.csv"), indices_train)
    save_list(os.path.join(result_dir_name, "indices_test.csv"), indices_test)

    save_dict(os.path.join(result_dir_name, 'mwe_dict.pkl'), mwe_dict)

    save_list(os.path.join(result_dir_name, 'mwe_list.csv'), mwe_list)


    save_list_of_lists('mwe_metadata.csv', mwe_metadata)


if __name__ == '__main__':
    main(sys.argv[1:])
