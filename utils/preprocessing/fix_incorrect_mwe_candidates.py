import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm

from utils.logging.logger import log_message, init_pandas_tqdm
from utils.tools.morfeusz import init_lemmatizer, lemmatize_single_mwe


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        help='path to the file containing data',
                        type=str)

    parser.add_argument('--mwe_list_path',
                        help='path to the list containing MWEs',
                        type=str)

    parser.add_argument('--output_path', help='path to output file', type=str)

    args = parser.parse_args()

    return args


def get_lemma_list(mwe_list, lemmatizer):
    return np.array([
        set(lemmatize_single_mwe(mwe, lemmatizer).split())
        for mwe in tqdm(mwe_list)
    ])


def get_words_set(mwe_list):
    words_2d_list = [[word for word in mwe.split(' ')] for mwe in mwe_list]

    words_set = set([item for sublist in words_2d_list for item in sublist])

    return words_set


def get_min_edit_neighbour(word, words_set):
    corrected_word = word
    min_dist = 4

    for candidate_word in words_set:
        if (word[:-2] == candidate_word[:-2] and
            (len(set(word[-2:]) - set(candidate_word[-2:])) +
             len(set(candidate_word[-2:]) - set(word[-2:]))) < min_dist):
            word = candidate_word

            min_dist = len(set(word[-2:]) - set(candidate_word[-2:])) + len(
                set(candidate_word[-2:]) - set(word[-2:]))

            if min_dist == 1:
                break

    return corrected_word


def get_mwe_min_edit_neighbour(mwe, words_set):
    return ' '.join(
        [get_min_edit_neighbour(word, words_set) for word in mwe.split(' ')])


def find_valid_mwe(mwe, lemmatized_mwe_list, mwe_list, words_set, lemmatizer):
    valid_mwe = mwe

    valid_mwe_found = False

    mwe_tuple = set(lemmatize_single_mwe(mwe, lemmatizer).split())

    for idx, valid_mwe_tuple in enumerate(lemmatized_mwe_list):
        if mwe_tuple.issubset(valid_mwe_tuple):
            if len(mwe_tuple) == len(valid_mwe_tuple):
                valid_mwe = mwe_list[idx]
            else:
                valid_mwe = 'part of longer mwe'

            valid_mwe_found = True
            break

    if not valid_mwe_found:
        valid_mwe = get_mwe_min_edit_neighbour(valid_mwe, words_set)
        valid_mwe_found = True

    return valid_mwe


def fix_incorrect_mwe_candidates(data_path, mwe_list_path, output_filepath,
                                 lemmatizer):
    data_df = pd.read_csv(data_path,
                          sep='\t',
                          header=None,
                          names=['measure_value', 'mwe_type', 'mwe'])

    mwe_df = pd.read_csv(mwe_list_path, sep='\t')

    mwe_list = mwe_df['lemma'].tolist()

    log_message('Lemmatizing MWE list...')
    lemmatized_mwe_list = get_lemma_list(mwe_list, lemmatizer)

    # get set containing all words in the dataset
    words_set = get_words_set(mwe_list)

    log_message(
        'Finding correct MWE substitution and checking for MWE inclusion...')
    data_df['corrected_mwe'] = data_df.progress_apply(
        lambda row: find_valid_mwe(row['mwe'], lemmatized_mwe_list, mwe_list,
                                   words_set, lemmatizer),
        axis=1)

    data_df = data_df.loc[data_df['corrected_mwe'] != 'part of longer mwe']

    data_df = pd.merge(data_df,
                       mwe_df[['lemma', 'is_correct']],
                       left_on='corrected_mwe',
                       right_on='lemma',
                       how='left')

    # mark not found MWEs with -1
    data_df['is_correct'] = data_df['is_correct'].fillna(-1.0)

    temp_columns = ['lemma', 'valid_mwe', 'index']

    data_df = data_df.loc[:, [
        col_name for col_name in data_df.columns
        if col_name not in temp_columns
    ]]

    data_df = data_df.drop_duplicates(subset=['mwe'], ignore_index=True)

    data_df.to_csv(output_filepath, sep='\t', index=False)


def main():
    args = parse_args()

    data_path = args.data_path
    mwe_list_path = args.mwe_list_path
    output_path = args.output_path

    init_pandas_tqdm()

    lemmatizer = init_lemmatizer()

    fix_incorrect_mwe_candidates(data_path, mwe_list_path, output_path,
                                 lemmatizer)


if __name__ == '__main__':
    main()
