import csv
import os
import pickle as pkl
import sys

from datetime import datetime

import fasttext
import numpy as np


def load_fasttext(model_path):
    model = fasttext.load_model(model_path)

    return model


def create_empty_file(filepath):
    with open(filepath, 'w') as f:
        column_names_line = '\t'.join(['mwe', 'fasttext_diff_vector', 'transformer_diff_vector'])
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


def load_list_of_lists(filepath):
    with open(filepath, 'r') as f:
        loaded_list_of_lists = np.array([np.array([elem for elem in row.strip().split(',')]) for row in f.readlines()])

        return loaded_list_of_lists


def list_of_lists_to_float(list_of_lists):
    converted_list_of_lists = np.array([np.array([float(elem) for elem in row]) for row in list_of_lists])

    return converted_list_of_lists


def load_list(filepath):
    with open(filepath, 'r') as f:
        loaded_list = np.array([elem for elem in f.readline().strip().split(',')])

        return loaded_list


def list_to_type(list_to_convert, type_func):
    converted_list = np.array([type_func(elem) for elem in list_to_convert])

    return converted_list


def load_dict(filepath):
    with open(filepath, 'rb') as f:
        loaded_dict = pkl.load(f)
    return loaded_dict


def get_fasttext_diff_vector(ft_model, mwe_words):
    first_word_emb = ft_model.get_word_vector(mwe_words[0])
    second_word_emb = ft_model.get_word_vector(mwe_words[1])

    return first_word_emb - second_word_emb


def generate_transformer_fasttext_embeddings(output_filepath, transformer_emb_list, indices_list, mwe_list, ft_model):
    create_empty_file(output_filepath)

    for ind, transfomer_emb in enumerate(transformer_emb_list):
        mwe = mwe_list[indices_list[ind]]

        if ' ' in mwe:
            mwe_words = mwe.split(' ')

        else:
            mwe_words = mwe.split('-')

        if len(mwe_words) != 2:
            continue

        fasttext_diff_vector = get_fasttext_diff_vector(ft_model, mwe_words)
        fasttext_diff_vector = [str(elem) for elem in fasttext_diff_vector]

        transformer_diff_vector = [str(elem) for elem in transfomer_emb]

        write_line_to_file(output_filepath,
                           '\t'.join([mwe, '\t'.join(fasttext_diff_vector), '\t'.join(transformer_diff_vector)]))

        if ind % 10000 == 0:
            print(f'{datetime.now().strftime("%H:%M:%S")}',
                  f'Processed {ind} samples')


def main(args):
    ft_model_path = "kgr10.plain.skipgram.dim300.neg10.bin"
    # ft_model_path = "kgr10.plain.cbow.dim300.neg10.bin"

    data_dir = 'transformer_embeddings_dataset'

    ft_model = load_fasttext(ft_model_path)

    print('Loading train data...')
    X_train = list_of_lists_to_float(load_list_of_lists(os.path.join(data_dir, "X_train.csv")))

    # get diff vector only
    X_train = np.array([embedding[768 * 2:] for embedding in X_train])

    print('Loading test data...')
    X_test = list_of_lists_to_float(load_list_of_lists(os.path.join(data_dir, "X_test.csv")))
    X_test = np.array([embedding[768 * 2:] for embedding in X_test])

    print('Loading indices files...')
    indices_train = list_to_type(load_list(os.path.join(data_dir, "indices_train.csv")), int)
    indices_test = list_to_type(load_list(os.path.join(data_dir, "indices_test.csv")), int)

    # print('Loading mwe dict...')
    # mwe_dict = load_dict(os.path.join(data_dir, 'mwe_dict.pkl'))

    print('Loading mwe list...')
    mwe_list = load_list(os.path.join(data_dir, 'mwe_list.csv'))

    # print('Loading mwe metadata...')
    # mwe_metadata = load_list_of_lists(os.path.join(data_dir, 'mwe_metadata.csv'))

    generate_transformer_fasttext_embeddings(data_dir + '/' + 'transformer_fasttext_diff_vectors_X_train.tsv', X_train,
                                             indices_train, mwe_list, ft_model)

    generate_transformer_fasttext_embeddings(data_dir + '/' + 'transformer_fasttext_diff_vectors_X_test.tsv', X_test,
                                             indices_test, mwe_list, ft_model)


if __name__ == '__main__':
    main(sys.argv[1:])
