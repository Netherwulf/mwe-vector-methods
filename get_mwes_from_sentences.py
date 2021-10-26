import sys

import numpy as np


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

        for line_idx, line in enumerate(content[1:]):
            line = line.strip()

            line_attributes = line.split('\t')

            mwe = line_attributes[0]
            is_correct = line_attributes[2]
            complete_mwe_in_sent = line_attributes[3]
            first_word_idx = line_attributes[4]
            first_word_orth = line_attributes[5]
            first_word_lemma = line_attributes[6]
            sentence = line_attributes[12]

            # complete MWE appears in the sentence
            if complete_mwe_in_sent == '1':
                write_line_to_file(complete_mwe_in_sent_output_file, '\t'.join(
                    [mwe, is_correct, first_word_idx, first_word_orth, first_word_lemma, sentence]))

            # only part of MWE appears in the sentence
            else:
                write_line_to_file(incomplete_mwe_in_sent_output_file, '\t'.join(
                    [mwe, is_correct, first_word_idx, first_word_orth, first_word_lemma, sentence]))

            if line_idx >= num_of_rows:
                break


def main(args):
    for filepath in args:
        get_mwes(filepath, 7 * 10 ** 6)


if __name__ == '__main__':
    main(sys.argv[1:])
