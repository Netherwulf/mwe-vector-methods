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


def find_valid_mwe(mwe, lemmatized_mwe_list, mwe_list, lemmatizer):
    valid_mwe = mwe

    mwe_tuple = set(lemmatize_single_mwe(mwe, lemmatizer).split())

    for idx, valid_mwe_tuple in enumerate(lemmatized_mwe_list):
        if mwe_tuple == valid_mwe_tuple:
            valid_mwe = mwe_list[idx]
            break

    return valid_mwe


def check_mwe_inclusion(mwe, lemmatized_mwe_list, lemmatizer):
    if '.' in mwe:
        return 0

    mwe_validity = 1

    mwe_tuple = set(lemmatize_single_mwe(mwe, lemmatizer).split())

    for valid_mwe_tuple in lemmatized_mwe_list:
        if (mwe_tuple.issubset(valid_mwe_tuple)
                and len(mwe_tuple) < len(valid_mwe_tuple)):
            mwe_validity = 0
            break

    return mwe_validity


def fix_incorrect_mwe_candidates(data_path, mwe_list_path, output_filepath,
                                 lemmatizer):
    data_df = pd.read_csv(data_path,
                          sep='\t',
                          header=None,
                          names=['measure_value', 'mwe_type', 'mwe'])

    data_df = data_df.drop_duplicates(subset=['mwe']).reset_index()

    mwe_df = pd.read_csv(mwe_list_path, sep='\t')

    mwe_list = mwe_df['lemma'].tolist()

    log_message('Lemmatizing MWE list...')
    lemmatized_mwe_list = get_lemma_list(mwe_list, lemmatizer)

    log_message('Finding correct MWE substitution...')
    data_df['corrected_mwe'] = data_df.progress_apply(
        lambda row: find_valid_mwe(row['mwe'], lemmatized_mwe_list, mwe_list,
                                   lemmatizer),
        axis=1)

    log_message('Checking MWE inclusion...')
    data_df['valid_mwe'] = data_df.progress_apply(
        lambda row: check_mwe_inclusion(row['mwe'], lemmatized_mwe_list,
                                        lemmatizer),
        axis=1)

    temp_columns = ['lemma', 'valid_mwe', 'index']

    data_df = data_df.loc[data_df['valid_mwe'] == 1, [
        col_name for col_name in data_df.columns
        if col_name not in temp_columns
    ]]

    data_df = pd.merge(data_df,
                       mwe_df[['lemma', 'is_correct']],
                       left_on='mwe',
                       right_on='lemma',
                       how='left')

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
