import os
import sys

import pandas as pd


def merge_sentence_lists(correct_mwe_filepath, incorrect_mwe_filepath, output_filepath):
    df_correct = pd.read_csv(correct_mwe_filepath, sep='\t')
    df_incorrect = pd.read_csv(incorrect_mwe_filepath, sep='\t')

    df_correct['is_correct'] = '1'
    df_incorrect['is_correct'] = '0'

    df_correct = df_correct.append(df_incorrect)

    df_correct.to_csv(output_filepath, sep='\t')


def main(args):
    correct_mwe_filepath = os.path.join('parseme_correct_mwes.tsv')
    incorrect_mwe_filepath = os.path.join('parseme_incorrect_mwes.tsv')
    output_filepath = os.path.join('parseme_merged_mwes.tsv')

    merge_sentence_lists(correct_mwe_filepath, incorrect_mwe_filepath, output_filepath)


if __name__ == '__main__':
    main(sys.argv[1:])
