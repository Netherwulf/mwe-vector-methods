import argparse
import os
import random

import numpy as np
import pandas as pd

from typing import List

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
        '--already_annotated_path',
        help='path to the file containing already annotated MWEs',
        type=str)

    parser.add_argument('--output_path', help='path to output file', type=str)

    args = parser.parse_args()

    return args


def load_candidates(filepath: str) -> pd.DataFrame:

    return pd.read_csv(filepath, sep='\t')


def load_annotated_mwes(filepath: str) -> List:
    annotations_df = pd.read_csv(filepath, sep='\t')

    return annotations_df['orth'].tolist()


def check_mwe_capitalization(mwe: str) -> int:
    mwe_components = mwe.split(' ')

    if '' not in mwe_components and all(
        [word[0].isupper() for word in mwe_components]):
        return 1
    else:
        return 0


def filter_candidates(candidates_df: pd.DataFrame, data_mwe_col_name: str,
                      annotated_mwes: np.ndarray, lemmatizer) -> pd.DataFrame:
    log_message('Lemmatizing candidate MWEs...')
    candidates_df['lemma'] = candidates_df.progress_apply(
        lambda row: lemmatize_single_mwe(row[data_mwe_col_name], lemmatizer),
        axis=1)

    log_message('Checking words capitalization...')
    candidates_df['is_capitalized'] = candidates_df.progress_apply(
        lambda row: check_mwe_capitalization(row[data_mwe_col_name]), axis=1)

    lemmatized_annotated_mwe = lemmatize_mwe_list(annotated_mwes, lemmatizer)

    filtered_candidates_df = candidates_df[
        ~(candidates_df['lemma'].isin(lemmatized_annotated_mwe))
        & (candidates_df['mwe_candidate_correctness'] == 1)
        & (candidates_df['is_capitalized'] == 0)]

    return filtered_candidates_df


def save_df_to_batches(df: pd.DataFrame,
                       output_path: str,
                       batch_size: int = 100) -> None:

    output_dir = output_path.split('/')[-1]

    idx_list = [idx for idx in range(len(df))]

    random.shuffle(idx_list)

    idx_subsets = [
        idx_list[i:i + batch_size] for i in range(0, len(idx_list), batch_size)
    ]

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for subset_idx, idx_subset in enumerate(idx_subsets):
        subset_df = df.iloc[idx_subset]

        subset_df.to_csv(os.path.join(output_path,
                                      f'{output_dir}_{subset_idx}.tsv'),
                         columns=['measure_value', 'mwe_type', 'mwe'],
                         sep='\t',
                         index=False)


def process_candidates_list(candidates_path: str, data_mwe_col_name: str,
                            annotated_mwes_path: str, output_path: str,
                            lemmatizer) -> None:
    log_message('Loading candidates df...')
    candidates_df = load_candidates(candidates_path)

    log_message('Loading annotated MWEs list...')
    annotated_mwes = load_annotated_mwes(annotated_mwes_path)

    log_message('Filtering MWE candidates...')
    filtered_candidates_df = filter_candidates(candidates_df,
                                               data_mwe_col_name,
                                               annotated_mwes, lemmatizer)

    log_message(f'Saving subsets to: {output_path}')

    save_df_to_batches(filtered_candidates_df, output_path, batch_size=100)


def main():
    init_pandas_tqdm()

    args = parse_args()

    data_path = args.data_path
    data_mwe_col_name = args.data_mwe_col_name
    already_annotated_path = args.already_annotated_path
    output_path = args.output_path

    lemmatizer = init_lemmatizer()

    process_candidates_list(data_path, data_mwe_col_name,
                            already_annotated_path, output_path, lemmatizer)


if __name__ == '__main__':
    main()
