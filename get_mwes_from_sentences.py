import sys

import morfeusz2
import numpy as np


# init Morfeusz2 lemmatizer
def init_lemmatizer():
    return morfeusz2.Morfeusz()  # initialize Morfeusz object


def create_empty_file(filepath):
    with open(filepath, 'w') as _:
        pass


def write_line_to_file(filepath, line):
    with open(filepath, 'a') as f:
        f.write(f'{line}\n')


def get_mwes(filepath, num_of_rows):
    filepath_name = filepath.split('/')[-1].split('.')[0]

    complete_mwe_in_sent_output_file = filepath_name + f'_mwe_list_complete_mwe_in_sent.tsv'
    create_empty_file(complete_mwe_in_sent_output_file)

    incomplete_mwe_in_sent_output_file = filepath_name + f'_mwe_list_incomplete_mwe_in_sent.tsv'
    create_empty_file(incomplete_mwe_in_sent_output_file)

    with open(filepath, 'r', errors='replace') as in_file:
        content = in_file.readlines()

        for line in content[1:]:
            line = line.strip()

            line_attributes = line.split('\t')

            mwe = line_attributes[0]
            is_correct = line_attributes[2]
            complete_mwe_in_sent = line_attributes[3]

            # complete MWE appears in the sentence
            if complete_mwe_in_sent == '1':
                write_line_to_file(complete_mwe_in_sent_output_file, '\t'.join(
                    [mwe, is_correct]))

            # only part of MWE appears in the sentence
            else:
                write_line_to_file(incomplete_mwe_in_sent_output_file, '\t'.join(
                    [mwe, is_correct]))


def main(args):
    for filepath in args:
        get_mwes(filepath, 4 * 10 ** 6)


if __name__ == '__main__':
    main(sys.argv[1:])
