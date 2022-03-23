import argparse
import datetime
import os
import sys

import pandas as pd


def log_message(message):
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : {message}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--correct_mwe_filepath',
                        help='filepath to correct mwe dataset',
                        type=str)

    parser.add_argument('--incorrect_mwe_filepath',
                        help='filepath to incorrect mwe dataset',
                        type=str)

    parser.add_argument('--output_filepath', help='output filepath', type=str)

    args = parser.parse_args()

    return args


def get_columns_dict(line_elems):
    return {line_elems[ind]: ind for ind in range(len(line_elems))}


def merge_datasets(correct_mwe_filepath, incorrect_mwe_filepath):
    log_message('Loading correct MWE file...')
    df_correct = pd.read_csv(correct_mwe_filepath, sep='\t')

    log_message('Loading incorrect MWE file...')
    df_incorrect = pd.read_csv(incorrect_mwe_filepath, sep='\t')

    df_correct['is_correct'] = '1'
    df_incorrect['is_correct'] = '0'

    df_correct = df_correct.append(df_incorrect)

    return df_correct


def merge_datasets_simple(correct_mwe_filepath, incorrect_mwe_filepath,
                          output_filepath):
    log_message('Loading correct MWE file...')
    with open(correct_mwe_filepath, 'r', encoding="utf-8",
              buffering=2000) as in_file, open(output_filepath,
                                               'a',
                                               encoding="utf-8",
                                               buffering=2000) as out_file:
        line_count = 0
        first_line = True

        for line in in_file:
            line_elems = line.rstrip().split('\t')

            if first_line:
                if 'is_correct' in line_elems:
                    label_column_idx = line_elems.index('is_correct')
                    line_elems = [
                        elem for elem in line_elems if elem != 'is_correct'
                    ]
                else:
                    label_column_idx = -1

                line_elems += ['is_correct']

                out_file.write('\t'.join(line_elems) + '\n')

                first_line = False

                continue

            else:
                line_count += 1

                if label_column_idx != -1:
                    line_elems = [
                        elem for i, elem in enumerate(line_elems)
                        if i != label_column_idx
                    ]

                line_elems += ['1']

                out_file.write('\t'.join(line_elems) + '\n')

            if line_count > 0 and line_count % 10000 == 0:
                log_message(f'Processed {line_count} lines')

            if line_count == 100000:
                break

    log_message('Loading incorrect MWE file...')
    with open(incorrect_mwe_filepath, 'r', encoding="utf-8",
              buffering=2000) as in_file, open(output_filepath,
                                               'a',
                                               encoding="utf-8",
                                               buffering=2000) as out_file:
        line_count = 0
        first_line = True

        for line in in_file:
            line_elems = line.rstrip().split('\t')

            if first_line:
                if 'is_correct' in line_elems:
                    label_column_idx = line_elems.index('is_correct')
                    line_elems = [
                        elem for elem in line_elems if elem != 'is_correct'
                    ]
                else:
                    label_column_idx = -1

                first_line = False

                continue

            else:
                line_count += 1

                if label_column_idx != -1:
                    line_elems = [
                        elem for i, elem in enumerate(line_elems)
                        if i != label_column_idx
                    ]

                line_elems += ['0']

                out_file.write('\t'.join(line_elems) + '\n')

            if line_count > 0 and line_count % 10000 == 0:
                log_message(f'Processed {line_count} lines')

            if line_count == 100000:
                break


def main():
    # dir_path = os.path.join('storage', 'parseme', 'en', 'preprocessed_data')

    # for dataset_type_idx, dataset_type in enumerate(['train', 'test',
    #                                                  'dev'][:-1]):
    #     correct_mwe_filepath = os.path.join(
    #         dir_path, dataset_type, f'parseme_{dataset_type}_correct_mwes.tsv')

    #     incorrect_mwe_filepath = os.path.join(
    #         dir_path, dataset_type,
    #         f'parseme_{dataset_type}_incorrect_mwes.tsv')

    #     if dataset_type_idx == 0:
    #         df = merge_sentence_lists(correct_mwe_filepath,
    #                                   incorrect_mwe_filepath)
    #         df['dataset_type'] = dataset_type

    #     else:
    #         curr_df = merge_sentence_lists(correct_mwe_filepath,
    #                                        incorrect_mwe_filepath)
    #         curr_df['dataset_type'] = dataset_type

    #         df = df.append(curr_df)
    args = parse_args()

    corr_mwe_filepath = args.correct_mwe_filepath
    incorr_mwe_filepath = args.incorrect_mwe_filepath
    out_filepath = args.output_filepath

    merge_datasets_simple(corr_mwe_filepath, incorr_mwe_filepath, out_filepath)

    # merged_df = merge_datasets(corr_mwe_filepath, incorr_mwe_filepath)

    # merged_df.to_csv(out_filepath, sep='\t', index=False)


if __name__ == '__main__':
    main()
