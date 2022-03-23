import argparse
import os

import numpy as np
import pandas as pd

from itertools import chain, combinations

from sklearn.metrics.pairwise import cosine_similarity

from datasets.fasttext.generate_fasttext_embeddings import load_fasttext
from utils.logging.logger import log_message
from utils.preprocessing.statistics.count_mwe_occurrences import init_lemmatizer, lemmatize_single_mwe, lemmatize_mwe_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        help='path to the file containing data',
                        type=str)

    parser.add_argument('--data_mwe_col_name',
                        help='column name containing MWE in data file',
                        type=str)

    parser.add_argument('--mwe_list_path',
                        help='path to the file MWE list',
                        type=str)

    parser.add_argument('--mwe_list_col_name',
                        help='column name containing MWE in MWE list file',
                        type=str)

    parser.add_argument('--output_path', help='path to output file', type=str)

    args = parser.parse_args()

    return args


def check_mwe_components(mwe, lemmatized_mwe_list, lemmatized_mwe_words_list,
                         lemmatizer) -> int:
    lemmatized_mwe = lemmatize_single_mwe(mwe, lemmatizer)

    lemmatized_mwe_words = lemmatized_mwe.split(' ')

    if (all([
            lemmatized_mwe_word in lemmatized_mwe_words_list
            for lemmatized_mwe_word in lemmatized_mwe_words
    ]) and lemmatized_mwe not in lemmatized_mwe_list):

        return 1

    else:

        return 0


def get_emb_diffs(mwe, ft_model) -> np.float64:
    mwe_words = mwe.split(' ')

    if len(mwe_words) < 2:
        return 0.0

    word_embs_list = np.array(
        [np.array(ft_model.get_word_vector(word)) for word in mwe_words])

    idx_list = list(range(len(word_embs_list)))

    idx_combinations = list(combinations(idx_list, 2))

    similarities_arr = np.array([0.0 for _ in range(len(idx_combinations))])

    for i, idx_combination in enumerate(idx_combinations):
        first_emb = word_embs_list[idx_combination[0]]
        second_emb = word_embs_list[idx_combination[1]]

        first_emb = first_emb.reshape((1, -1))
        second_emb = second_emb.reshape((1, -1))

        similarities_arr[i] = cosine_similarity(first_emb, second_emb)[0][0]

    return np.mean(similarities_arr, dtype=np.float32)


def get_good_incorrect_mwe_candidates(data_path, data_mwe_col_name,
                                      mwe_list_path, mwe_list_col_name,
                                      output_path, lemmatizer, ft_model):
    log_message('loading data...')

    df = pd.read_csv(data_path,
                     sep='\t',
                     header=None,
                     names=['combined_value', 'mwe_type', 'mwe'])

    mwe_list = pd.read_csv(mwe_list_path, sep='\t')[mwe_list_col_name].tolist()

    mwe_words = list(chain.from_iterable([mwe.split(' ') for mwe in mwe_list]))

    log_message(f'lemmatizing data...')

    lemmatized_mwe_list = lemmatize_mwe_list(mwe_list, lemmatizer)

    lemmatized_mwe_words = lemmatize_mwe_list(mwe_words, lemmatizer)

    log_message('checking components validity...')

    df['valid_components'] = df.apply(lambda row: check_mwe_components(
        row[data_mwe_col_name], lemmatized_mwe_list, lemmatized_mwe_words,
        lemmatizer),
                                      axis=1)

    log_message('calculating cosine similarity...')

    df['cosine_similarity'] = df.apply(
        lambda row: get_emb_diffs(row[data_mwe_col_name], ft_model), axis=1)

    df.loc[:, 'cosine_similarity'] = df['cosine_similarity'].round(4)

    df['mwe_candidate_score'] = df['valid_components'] * df['cosine_similarity']

    df = df[(df['cosine_similarity'] > 0.0) & (df['cosine_similarity'] < 1.0)]

    log_message('saving df to tsv...')

    df = df.sort_values(by=['mwe_candidate_score', 'cosine_similarity'],
                        ascending=[False, False])

    df.to_csv(output_path, sep='\t', index=False)


def main():
    args = parse_args()

    data_path = args.data_path
    data_mwe_col_name = args.data_mwe_col_name
    mwe_list_path = args.mwe_list_path
    mwe_list_col_name = args.mwe_list_col_name
    output_path = args.output_path

    lemmatizer = init_lemmatizer()

    ft_model_path = os.path.join('storage', 'pretrained_models',
                                 'kgr10.plain.skipgram.dim300.neg10.bin')

    ft_model = load_fasttext(ft_model_path)

    get_good_incorrect_mwe_candidates(data_path, data_mwe_col_name,
                                      mwe_list_path, mwe_list_col_name,
                                      output_path, lemmatizer, ft_model)


if __name__ == '__main__':
    main()
