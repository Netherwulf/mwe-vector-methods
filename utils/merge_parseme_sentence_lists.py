import os
import sys

import pandas as pd


def merge_sentence_lists(correct_mwe_filepath, incorrect_mwe_filepath):
    df_correct = pd.read_csv(correct_mwe_filepath, sep='\t')
    df_incorrect = pd.read_csv(incorrect_mwe_filepath, sep='\t')

    df_correct['is_correct'] = '1'
    df_incorrect['is_correct'] = '0'

    df_correct = df_correct.append(df_incorrect)

    return df_correct


def main(args):
    dir_path = os.path.join('storage', 'parseme', 'en', 'preprocessed_data')

    for dataset_type_idx, dataset_type in enumerate(['train', 'test',
                                                     'dev'][:-1]):
        correct_mwe_filepath = os.path.join(
            dir_path, dataset_type, f'parseme_{dataset_type}_correct_mwes.tsv')

        incorrect_mwe_filepath = os.path.join(
            dir_path, dataset_type,
            f'parseme_{dataset_type}_incorrect_mwes.tsv')

        if dataset_type_idx == 0:
            df = merge_sentence_lists(correct_mwe_filepath,
                                      incorrect_mwe_filepath)
            df['dataset_type'] = dataset_type

        else:
            curr_df = merge_sentence_lists(correct_mwe_filepath,
                                           incorrect_mwe_filepath)
            curr_df['dataset_type'] = dataset_type

            df = df.append(curr_df)

    df.to_csv(os.path.join(dir_path, 'parseme_merged_data.tsv'),
              sep='\t',
              index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
