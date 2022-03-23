import argparse
import datetime
import os
import sys

import pandas as pd


def log_message(message):
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : {message}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath',
                        help='path to the file containing data',
                        type=str)
    parser.add_argument('--mwe_col_idx',
                        help='index of column containing MWE',
                        type=int)
    parser.add_argument('--class_col_idx',
                        help='index of column containing class',
                        type=int)
    parser.add_argument('--col_count',
                        help='number of columns in file',
                        type=int)
    parser.add_argument('--class_to_undersample',
                        help='class to undersample',
                        type=int)
    parser.add_argument('--undersampling_ratio',
                        help='undersampled class to other classes ratio',
                        type=float)

    args = parser.parse_args()

    return args


def undersample_mwe_occurrences(filepath, occurrences_num, mwe_col_ind,
                                class_col_ind, col_count, log_batch_size,
                                undersampled_class):
    filename = filepath.split('/')[-1].split('.')[0]
    dir_path = os.path.join(*filepath.split('/')[:-1])

    output_path = os.path.join(dir_path, f'{filename}_undersampled.tsv')

    mwe_dict = {}
    line_count = 0

    with open(filepath, 'r',
              buffering=2000) as in_file, open(output_path,
                                               'a',
                                               buffering=2000) as out_file:
        for line in in_file:

            if line_count == 0:
                out_file.write(line)

                line_count += 1

            else:
                line_attributes = line.split('\t')

                line_count += 1

                if len(line_attributes) == col_count:
                    mwe = line_attributes[mwe_col_ind]

                    if class_col_ind is not None and line_attributes[
                            class_col_ind] != undersampled_class:
                        out_file.write(line)
                        continue

                    if (mwe in mwe_dict.keys()
                            and mwe_dict[mwe] == occurrences_num):
                        continue

                    if mwe not in mwe_dict.keys():
                        mwe_dict[mwe] = 1

                    else:
                        mwe_dict[mwe] += 1

                    out_file.write(line)

            if line_count != 0 and line_count % log_batch_size == 0:
                print(f'Processed {line_count} lines')


def undersample_class_simple(filepath, class_to_undersample,
                             undersampling_ratio):
    filename = filepath.split('/')[-1].split('.')[0]
    dir_path = os.path.join(*filepath.split('/')[:-1])
    output_path = os.path.join(
        dir_path, f'{filename}_undersampled_ratio_{undersampling_ratio}.tsv')

    df = pd.read_csv(filepath, sep='\t', on_bad_lines='skip')

    result_df = pd.DataFrame(columns=df.columns)

    for dataset_type in ['train', 'dev', 'test']:
        df_not_to_undersample = df[(df['is_correct'] != class_to_undersample)
                                   & (df['dataset_type'] == dataset_type)]
        df_to_undersample = df[(df['is_correct'] == class_to_undersample)
                               & (df['dataset_type'] == dataset_type)]

        df_undersampled = df_to_undersample.sample(n=int(
            undersampling_ratio * len(df_not_to_undersample)),
                                                   random_state=1)

        result_df = result_df.append(df_not_to_undersample)
        result_df = result_df.append(df_undersampled)

    print(
        'Dataframe class counts:',
        f'train - 0: {len(result_df[(result_df["is_correct"] == 0) & (result_df["dataset_type"] == "train")])}',
        f'train - 1: {len(result_df[(result_df["is_correct"] == 1) & (result_df["dataset_type"] == "train")])}',
        f'dev - 0: {len(result_df[(result_df["is_correct"] == 0) & (result_df["dataset_type"] == "dev")])}',
        f'dev - 1: {len(result_df[(result_df["is_correct"] == 1) & (result_df["dataset_type"] == "dev")])}',
        f'test - 0: {len(result_df[(result_df["is_correct"] == 0) & (result_df["dataset_type"] == "test")])}',
        f'test - 1: {len(result_df[(result_df["is_correct"] == 1) & (result_df["dataset_type"] == "test")])}',
        sep='\n')

    result_df.to_csv(output_path, sep='\t', index=False)


def main():
    args = parse_args()

    filepath = args.filepath
    # max_mwe_occurrences = 10
    # mwe_col_idx = args.mwe_col_idx  # 7
    # class_col_idx = args.class_col_idx  # 10
    # col_count = args.col_count  # 12
    # log_batch_size = 10000
    class_to_undersample = args.class_to_undersample  # 0
    undersampling_ratio = args.undersampling_ratio  # 1.2

    # for filepath in args:
    # undersample_mwe_occurrences(filepath, max_mwe_occurrences, mwe_col_idx,
    #                             class_col_idx, col_count, log_batch_size,
    #                             class_to_undersample)
    undersample_class_simple(filepath, class_to_undersample,
                             undersampling_ratio)


if __name__ == '__main__':
    main()
