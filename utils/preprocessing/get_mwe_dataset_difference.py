import argparse

import pandas as pd

from utils.logging.logger import log_message


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--first_list_path',
                        help='path to the first file containing data',
                        type=str)

    parser.add_argument('--second_list_path',
                        help='path to the second file containing data',
                        type=str)

    parser.add_argument('--output_path', help='path to output file', type=str)

    args = parser.parse_args()

    return args


def get_mwe_dataset_difference(first_list_path, second_list_path,
                               output_path) -> None:
    first_df = pd.read_csv(first_list_path, sep='\t')
    second_df = pd.read_csv(second_list_path, sep='\t')

    second_mwe_list = second_df['lemma'].tolist()

    diff_df = first_df[~first_df['lemma'].isin(second_mwe_list)]

    log_message(f'first_df length: {len(first_df)}',
                f'second df length: {len(second_df)}',
                f'diff df length: {len(diff_df)}')

    diff_df.to_csv(output_path, sep='\t', index=False)


def main():
    args = parse_args()

    first_list_path = args.first_list_path
    second_list_path = args.second_list_path
    output_path = args.output_path

    get_mwe_dataset_difference(first_list_path, second_list_path, output_path)


if __name__ == '__main__':
    main()
