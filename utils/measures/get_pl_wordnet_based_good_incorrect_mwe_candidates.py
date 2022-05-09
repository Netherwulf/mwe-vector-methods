import argparse

import numpy as np
import pandas as pd

from itertools import chain
from typing import List

from tqdm import tqdm

from utils.logging.logger import log_message, init_pandas_tqdm
from utils.tools.morfeusz import (init_lemmatizer, lemmatize_single_mwe,
                                  lemmatize_mwe_list)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        help='path to the file containing data',
                        type=str)

    parser.add_argument('--data_mwe_col_name',
                        help='column name containing MWE in data file',
                        type=str)

    parser.add_argument(
        '--pl_wordnet_words_path',
        help='path to the file containing words from plWordnet (Słowosieć)',
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


def load_pl_wordnet_words(filepath: str) -> np.ndarray:
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        single_words = [
            line.strip('\n') for line in tqdm(lines) if '"' not in line
        ]

        expressions_list = [
            line.strip('\n').strip('"').split(' ') for line in tqdm(lines)
            if '"' in line
        ]

        expressions_words = [
            word for expression_words in tqdm(expressions_list)
            for word in expression_words
        ]

        return np.array(single_words + expressions_words)


def check_mwe_candidate_correctness(mwe: str,
                                    lemmatized_pl_wordnet_words: List[str],
                                    lemmatized_mwe_words_list: List[str],
                                    lemmatized_mwe_list: List[str],
                                    lemmatizer) -> int:
    """
    Check for each MWE candidate if:
    - EACH component exists in plWordnet (Słowosieć)
    - AT LEAST ONE component exists in any (correct/incorrect) MWE from the dictionary
    - COMPLETE MWE candidate does not exist in the MWE dictionary
    """
    lemmatized_mwe = lemmatize_single_mwe(mwe, lemmatizer)

    lemmatized_mwe_words = lemmatized_mwe.split(' ')

    if (all([
            lemmatized_mwe_word in lemmatized_pl_wordnet_words
            for lemmatized_mwe_word in lemmatized_mwe_words
    ]) and any([
            lemmatized_mwe_word in lemmatized_mwe_words_list
            for lemmatized_mwe_word in lemmatized_mwe_words
    ]) and lemmatized_mwe not in lemmatized_mwe_list):

        return 1

    else:

        return 0


def get_good_incorrect_mwe_candidates(data_path, data_mwe_col_name,
                                      pl_wordnet_words_path, mwe_list_path,
                                      mwe_list_col_name, output_path,
                                      lemmatizer):
    log_message('loading data...')

    df = pd.read_csv(data_path,
                     sep='\t',
                     usecols=['measure_value', 'mwe_type', 'corrected_mwe'])

    pl_wordnet_words = load_pl_wordnet_words(pl_wordnet_words_path)

    mwe_list = pd.read_csv(mwe_list_path, sep='\t')[mwe_list_col_name].tolist()

    mwe_words = list(chain.from_iterable([mwe.split(' ') for mwe in mwe_list]))

    log_message(f'lemmatizing data...')

    lemmatized_pl_wordnet_words = lemmatize_mwe_list(pl_wordnet_words,
                                                     lemmatizer)

    lemmatized_mwe_list = lemmatize_mwe_list(mwe_list, lemmatizer)

    lemmatized_mwe_words = lemmatize_mwe_list(mwe_words, lemmatizer)

    log_message('checking components validity...')

    df['mwe_candidate_correctness'] = df.progress_apply(
        lambda row: check_mwe_candidate_correctness(
            row[data_mwe_col_name], lemmatized_pl_wordnet_words,
            lemmatized_mwe_words, lemmatized_mwe_list, lemmatizer),
        axis=1)

    log_message('saving df to tsv...')

    df = df.rename(columns={'corrected_mwe': 'mwe'})

    df = df.sort_values(by=['mwe_candidate_correctness'], ascending=[False])

    df.to_csv(output_path, sep='\t', index=False)


def main():
    init_pandas_tqdm()

    args = parse_args()

    data_path = args.data_path
    data_mwe_col_name = args.data_mwe_col_name
    pl_wordnet_words_path = args.pl_wordnet_words_path
    mwe_list_path = args.mwe_list_path
    mwe_list_col_name = args.mwe_list_col_name
    output_path = args.output_path

    lemmatizer = init_lemmatizer()

    get_good_incorrect_mwe_candidates(data_path, data_mwe_col_name,
                                      pl_wordnet_words_path, mwe_list_path,
                                      mwe_list_col_name, output_path,
                                      lemmatizer)


if __name__ == '__main__':
    main()
