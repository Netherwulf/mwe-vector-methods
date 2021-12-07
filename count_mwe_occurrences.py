import csv
import string
import sys

import numpy as np

from datetime import datetime
from typing import List

import morfeusz2


# read MWEs from Słowosieć tsv or csv file
def read_mwe(filepath, separator, column_index) -> []:
    with open(filepath, 'r', encoding="utf-8") as f:
        content = list(csv.reader(f, delimiter=separator, quotechar='"'))
        mwe_list = [sublist[column_index] for sublist in content[1:] if len(sublist) != 0]
        mwe_list = [mwe for mwe in mwe_list if not any(character.isupper() for character in mwe)]
        return mwe_list


# init Morfeusz2 lemmatizer
def init_lemmatizer():
    return morfeusz2.Morfeusz()  # initialize Morfeusz object


# lemmatize MWEs
def lemmatize_mwe(mwe_list, lemmatizer) -> List[str]:
    lemmatized_mwe_list = ['*' * 200 for _ in range(len(mwe_list))]

    for i, mwe in enumerate(mwe_list):
        mwe_words = [token for token in mwe.split(' ')]
        lemmatized_mwe_list[i] = ' '.join(
            [str(lemmatizer.analyse(word)[0][2][1]) if word not in string.punctuation else word for word in mwe_words])

    return lemmatized_mwe_list


def clean_mwe_list(mwe_list, lemmatized_mwe_list):
    for i, mwe in enumerate(mwe_list):
        if '\xa0' in mwe:
            mwe_list[i] = mwe.replace('\xa0', ' ')
            lemmatized_mwe_list[i] = mwe.replace('\xa0', ' ')

    mwe_list = [mwe for mwe in mwe_list if len(mwe.split(' ')) == 2]
    lemmatized_mwe_list = [mwe for mwe in lemmatized_mwe_list if len(mwe.split(' ')) == 2]

    return mwe_list, lemmatized_mwe_list


def load_mwes(mwe_file):
    lemmatizer = init_lemmatizer()

    mwe_list = read_mwe(mwe_file, '\t', 2)
    lemmatized_mwes = lemmatize_mwe(mwe_list, lemmatizer)

    mwe_list, lemmatized_mwes = clean_mwe_list(mwe_list, lemmatized_mwes)

    return mwe_list, lemmatized_mwes


def count_mwe_occurrences(mwe_list, lemmatized_mwe_list, filepath):
    mwe_dict = {lemmatized_mwe: 0 for lemmatized_mwe in lemmatized_mwe_list}

    with open(filepath, 'r', encoding='utf-8') as in_file:
        line_idx = 0

        # skip first line
        line = in_file.readline()

        while line:
            line = in_file.readline()

            lemmatized_mwe = line.split('\t')[1]

            mwe_dict[lemmatized_mwe] += 1

            if line_idx % 100000 == 0 and line_idx > 0:
                print(f'{datetime.now().time()} - Processed {line_idx} files')

            line_idx += 1

        mwe_occurrences = [occurrence_count for occurrence_count in mwe_dict.values()]
        empty_mwe_dict = [mwe for mwe in mwe_dict.keys() if mwe_dict[mwe] == 0]

        print(f'{datetime.now().time()} - Results for file: {filepath}',
              f'Max occurrences: {np.max(mwe_occurrences)}',
              f'Median occurrences: {np.median(mwe_occurrences)}',
              f'Mean occurrences: {np.mean(mwe_occurrences)}',
              f'Min occurrences: {np.min(mwe_occurrences)}',
              f'Zero occurrences count: {len(empty_mwe_dict)}',
              sep='\n')


def main(args):
    mwes_filepath = 'scaled_vector_association_measure_correct_mwe_best_f1.tsv'

    print(f'{datetime.now().time()} - Reading MWE files and lemmatizing...')

    mwe_list, lemmatized_mwes = load_mwes(mwes_filepath)

    print(f'{datetime.now().time()} - Finished reading MWE files and lemmatizing...')

    for filepath in args:
        count_mwe_occurrences(mwe_list, lemmatized_mwes, filepath)


if __name__ == '__main__':
    main(sys.argv[1:])
