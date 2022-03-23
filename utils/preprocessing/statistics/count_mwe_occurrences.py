import argparse
import string

import numpy as np

import datetime
from typing import List

import morfeusz2


def log_message(message):
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : {message}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        help='path to the file containing sentences',
                        type=str)
    parser.add_argument('--data_mwe_col_idx',
                        help='index of column containing MWE in sentence_file',
                        type=int)
    parser.add_argument('--mwe_list_path', help='path to MWE list', type=str)
    parser.add_argument('--mwe_col_idx',
                        help='index of column containing MWE in MWE list',
                        type=int)

    args = parser.parse_args()

    return args


# read MWEs from Słowosieć tsv or csv file
# def read_mwe(filepath, separator, column_index) -> List[str]:
#     with open(filepath, 'r', encoding="utf-8") as f:
#         content = list(csv.reader(f, delimiter=separator, quotechar='"'))
#         mwe_list = [
#             sublist[column_index] for sublist in content[1:]
#             if len(sublist) != 0
#         ]
#         mwe_list = [
#             mwe for mwe in mwe_list
#             if not any(character.isupper() for character in mwe)
#         ]
#         return mwe_list


# init Morfeusz2 lemmatizer
def init_lemmatizer():
    return morfeusz2.Morfeusz()  # initialize Morfeusz object


# lemmatize single MWE
def lemmatize_single_mwe(mwe, lemmatizer) -> string:

    mwe_words = [token for token in mwe.split(' ')]
    lemmatized_mwe = ' '.join([
        str(lemmatizer.analyse(word)[0][2][1])
        if word not in string.punctuation else word for word in mwe_words
    ])

    return lemmatized_mwe


# lemmatize MWEs
def lemmatize_mwe_list(mwe_list, lemmatizer) -> List[str]:
    lemmatized_mwe_list = ['*' * 200 for _ in range(len(mwe_list))]

    for i, mwe in enumerate(mwe_list):
        lemmatized_mwe_list[i] = lemmatize_single_mwe(mwe, lemmatizer)

    return lemmatized_mwe_list


def clean_mwe_list(mwe_list, lemmatized_mwe_list):
    for i, mwe in enumerate(mwe_list):
        if '\xa0' in mwe:
            mwe_list[i] = mwe.replace('\xa0', ' ')
            lemmatized_mwe_list[i] = mwe.replace('\xa0', ' ')

    mwe_list = [mwe for mwe in mwe_list if len(mwe.split(' ')) == 2]
    lemmatized_mwe_list = [
        mwe for mwe in lemmatized_mwe_list if len(mwe.split(' ')) == 2
    ]

    return mwe_list, lemmatized_mwe_list


# def load_mwes(mwe_file):
#     lemmatizer = init_lemmatizer()

#     mwe_list = read_mwe(mwe_file, '\t', 2)
#     lemmatized_mwes = lemmatize_mwe(mwe_list, lemmatizer)

#     mwe_list, lemmatized_mwes = clean_mwe_list(mwe_list, lemmatized_mwes)

#     return mwe_list, lemmatized_mwes


def read_mwe(mwe_filepath,
             column_index,
             lemmatizer,
             separator='\t') -> List[str]:
    mwe_list = []

    line_idx = 0

    with open(mwe_filepath, 'r', encoding="utf-8") as f:
        for line in f:

            # skip first line
            if line_idx == 0:
                line_idx += 1
                continue

            line = line.strip()
            line_elems = line.split(separator)

            # check if line is empty
            if len(line_elems) < 2:
                line_idx += 1
                continue

            mwe_list.append(line_elems[column_index])

            line_idx += 1

        lemmatized_mwe_list = lemmatize_mwe_list(mwe_list, lemmatizer)

    return lemmatized_mwe_list


def count_mwe_occurrences(mwe_list,
                          data_filepath,
                          data_mwe_col_idx,
                          lemmatizer,
                          separator='\t'):
    mwe_dict = {lemmatized_mwe: 0 for lemmatized_mwe in mwe_list}

    occurrence_count = 0

    with open(data_filepath, 'r', encoding='utf-8') as in_file:

        for line in in_file:

            # skip first line
            if occurrence_count == 0:
                occurrence_count += 1
                continue

            line = line.strip()

            line_elems = line.split(separator)

            if len(line_elems) < 2:
                continue

            mwe = line_elems[data_mwe_col_idx]

            lemmatized_mwe = lemmatize_single_mwe(mwe, lemmatizer)

            if lemmatized_mwe not in mwe_dict.keys():
                continue

            else:
                mwe_dict[lemmatized_mwe] += 1

            if occurrence_count % 10000 == 0 and occurrence_count > 0:
                log_message(f'Processed {occurrence_count} files')

            occurrence_count += 1

        mwe_occurrences = [
            occurrence_count for occurrence_count in mwe_dict.values()
        ]

        empty_mwe_dict = [mwe for mwe in mwe_dict.keys() if mwe_dict[mwe] == 0]

        print(f'Results for file: {data_filepath}',
              f'Max occurrences: {np.max(mwe_occurrences)}',
              f'Median occurrences: {np.median(mwe_occurrences)}',
              f'Mean occurrences: {np.mean(mwe_occurrences)}',
              f'Min occurrences: {np.min(mwe_occurrences)}',
              f'Zero occurrences count: {len(empty_mwe_dict)}',
              f'Total MWE count: {len(mwe_list)}',
              f'Total MWE occurrences count: {sum(mwe_occurrences)}',
              sep='\n')


def main():
    args = parse_args()

    data_path = args.data_path
    data_mwe_col_idx = args.data_mwe_col_idx
    mwe_list_path = args.mwe_list_path
    mwe_col_idx = args.mwe_col_idx

    lemmatizer = init_lemmatizer()

    # mwes_filepath = 'scaled_vector_association_measure_correct_mwe_best_f1.tsv'

    log_message('Reading MWE files...')

    # mwe_list, lemmatized_mwes = load_mwes(mwes_filepath)

    mwe_list = read_mwe(mwe_list_path, mwe_col_idx, lemmatizer)

    log_message('Finished reading MWE files...')

    count_mwe_occurrences(mwe_list, data_path, data_mwe_col_idx, lemmatizer)


if __name__ == '__main__':
    main()
