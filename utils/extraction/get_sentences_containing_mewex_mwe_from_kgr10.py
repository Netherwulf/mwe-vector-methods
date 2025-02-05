import csv
import os
import string
import sys
import xml.etree.ElementTree as ET

from datetime import datetime
from glob import glob
from typing import List, Tuple

import morfeusz2


# find morph.xml.ccl.xml files in dir recursively
def find_xml(dir_path) -> List[str]:
    return [
        filepath
        for filepath in glob(dir_path + '/**/*.ccl.xml', recursive=True)
    ]


# read MWEs from Słowosieć tsv or csv file
def read_mwe(filepath, separator, column_index) -> List:
    with open(filepath, 'r', encoding="utf-8") as f:
        content = list(csv.reader(f, delimiter=separator, quotechar='"'))
        mwe_list = [
            sublist[column_index] for sublist in content[1:]
            if len(sublist) != 0
        ]
        mwe_list = [
            mwe for mwe in mwe_list
            if not any(character.isupper() for character in mwe)
        ]
        return mwe_list


# read plain text from XML file
def read_xml(filepath) -> Tuple[List, List]:
    tree = ET.parse(filepath)
    sentences = tree.findall('.//sentence')

    orths = []
    lemmas = []

    for sentence in sentences:
        sentence_orths = []
        sentence_lemmas = []

        for sentence_token in sentence:
            for token_attr in sentence_token:
                if token_attr.tag == 'orth':
                    sentence_orths.append(token_attr.text)

                elif token_attr.tag == 'lex':
                    for lex_attr in token_attr:
                        if lex_attr.tag == 'base':
                            sentence_lemmas.append(lex_attr.text)

        orths.append(sentence_orths)
        lemmas.append(sentence_lemmas)

    return orths, lemmas


# init Morfeusz2 lemmatizer
def init_lemmatizer():
    return morfeusz2.Morfeusz()  # initialize Morfeusz object


# lemmatize MWEs
def lemmatize_mwe(mwe_list, lemmatizer) -> List[str]:
    lemmatized_mwe_list = ['*' * 200 for _ in range(len(mwe_list))]

    for i, mwe in enumerate(mwe_list):
        mwe_words = [token for token in mwe.split(' ')]
        lemmatized_mwe_list[i] = ' '.join([
            str(lemmatizer.analyse(word)[0][2][1])
            if word not in string.punctuation else word for word in mwe_words
        ])

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


def load_mwes(mwe_file):
    lemmatizer = init_lemmatizer()

    mwe_list = read_mwe(mwe_file, '\t', 2)
    lemmatized_mwes = lemmatize_mwe(mwe_list, lemmatizer)

    mwe_list, lemmatized_mwes = clean_mwe_list(mwe_list, lemmatized_mwes)

    return mwe_list, lemmatized_mwes


def get_restricted_words_list():
    return [
        'się', 'ja', 'ten', 'od', 'do', 'bez', 'beze', 'chyba', 'co', 'dla',
        'dzięki', 'dziela', 'gwoli', 'jako', 'kontra', 'krom', 'ku', 'miast',
        'na', 'nad', 'nade', 'naokoło', 'o', 'z', 'po', 'w', 'we', 'być'
    ] + list(string.punctuation)


def create_empty_file(filepath):
    with open(filepath, 'w') as _:
        pass


def write_line_to_file(filepath, line):
    with open(filepath, 'a') as f:
        f.write(f'{line}\n')


def write_new_samples_to_file(output_file, matched_mwe_list, mwe_orth_list,
                              lemmatized_mwe_list, is_complete_mwe,
                              first_word_index, first_word_orth,
                              first_word_lemma, second_word_index,
                              second_word_orth, second_word_lemma, dir_index,
                              file_index, sentence):
    is_complete_mwe_value = '1' if is_complete_mwe else '0'
    second_word_index_value = str(
        second_word_index) if is_complete_mwe else '-1'

    for mwe_ind, matched_mwe_lemma in enumerate(matched_mwe_list):
        if mwe_ind > 0 and matched_mwe_lemma == matched_mwe_list[mwe_ind - 1]:
            continue

        mwe = mwe_orth_list[lemmatized_mwe_list.index(matched_mwe_lemma)]
        write_line_to_file(
            output_file, '\t'.join([
                mwe, matched_mwe_lemma, is_complete_mwe_value,
                str(first_word_index), first_word_orth, first_word_lemma,
                second_word_index_value, second_word_orth, second_word_lemma,
                str(dir_index),
                str(file_index), sentence
            ]))


def get_sentences_containing_mwe(output_file, mwe_list, lemmatized_mwes,
                                 sentences_orths, sentences_lemmas,
                                 restricted_words_list, dir_index, file_index):
    for sentence_ind, sentence in enumerate(sentences_lemmas):
        if len(sentences_orths[sentence_ind]) != len(
                sentences_lemmas[sentence_ind]):
            # print(f'orths: {sentences_orths[sentence_ind]}',
            #       f'lemmas: {sentence}',
            #       sep='\n')
            cleaned_sentence = [
                word for i, word in enumerate(sentences_lemmas[sentence_ind])
                if i > 0 and
                sentences_lemmas[sentence_ind][i - 1].lower() != word.lower()
            ]
            sentence = cleaned_sentence

            # if the sentence can't be simply repaired by removing duplicated word then skip the sentence
            if len(sentence) != len(sentences_orths[sentence_ind]):
                continue

        if '/' in sentence:
            continue

        word_in_complete_mwe_list = [False for _ in range(len(sentence))]

        for lemma_ind, lemma in enumerate(sentence):
            # if a word is detected as a part of MWE then don't classify it as an occurrence of another MWE
            # if lemma in restricted_words_list or word_in_complete_mwe_list[lemma_ind]:
            #     continue

            # if word is a restricted word or contains digits, then skip it
            if lemma in restricted_words_list or any(char.isdigit()
                                                     for char in lemma):
                continue

            # check if word is part of complete MWE occurring in the sentence
            if not word_in_complete_mwe_list[lemma_ind] and lemma_ind != len(
                    sentence) - 1:
                matching_mwe_list = [
                    corr_mwe for corr_mwe in lemmatized_mwes
                    if ' '.join([sentence[lemma_ind], sentence[lemma_ind +
                                                               1]]) == corr_mwe
                ]

                if len(matching_mwe_list) != 0:
                    word_in_complete_mwe_list[lemma_ind] = True
                    word_in_complete_mwe_list[lemma_ind + 1] = True

                mwe_orths_list = mwe_list
                lemmatized_mwe_list = lemmatized_mwes

                write_new_samples_to_file(
                    output_file, matching_mwe_list, mwe_orths_list,
                    lemmatized_mwe_list, True, str(lemma_ind),
                    sentences_orths[sentence_ind][lemma_ind],
                    sentence[lemma_ind], str(lemma_ind + 1),
                    sentences_orths[sentence_ind][lemma_ind + 1],
                    sentence[lemma_ind + 1], dir_index, file_index,
                    ' '.join(sentences_orths[sentence_ind]))

            # if word didn't occur in any complete MWE then
            # check if word occurs in sentence even if whole MWE doesn't occur in it
            if not word_in_complete_mwe_list[lemma_ind]:
                matching_mwe_list = [
                    corr_mwe for corr_mwe in lemmatized_mwes
                    if lemma == corr_mwe.split(' ')[0]
                    or lemma == corr_mwe.split(' ')[1]
                ]

                mwe_orths_list = mwe_list
                lemmatized_mwe_list = lemmatized_mwes

                write_new_samples_to_file(
                    output_file, matching_mwe_list, mwe_orths_list,
                    lemmatized_mwe_list, False, str(lemma_ind),
                    sentences_orths[sentence_ind][lemma_ind],
                    sentence[lemma_ind], '-1', 'null', 'null', dir_index,
                    file_index, ' '.join(sentences_orths[sentence_ind]))


def main(args):
    mwes_list_dir = os.path.join('..', '..', 'storage', 'kgr10', 'mewex')
    mwes_file = 'scaled_vector_association_measure_correct_mwe_best_f1_with_corrected_mwes_pl_wordnet_based_filtering.tsv'
    mwes_filepath = os.path.join(mwes_list_dir, mwes_file)

    output_dir = os.path.join('..', '..', 'storage',
                              'kgr10_containing_mewex_results', 'raw_data')
    base_output_file_name = 'kgr10_sentences_containing_MeWeX_mwe.tsv'

    print(f'{datetime.now().time()} - Reading MWE files and lemmatizing...')

    mwe_list, lemmatized_mwes = load_mwes(mwes_filepath)

    print(
        f'{datetime.now().time()} - Finished reading MWE files and lemmatizing...'
    )

    restricted_words_list = get_restricted_words_list()

    # args is list of directories to search MWE occurrences inside
    for dir_index, dir_path in enumerate(args):
        print(f'{datetime.now().time()} - Reading files from {dir_path}')
        output_file = f'{base_output_file_name.split(".")[0]}_{dir_path.split("/")[-1]}.tsv'
        output_filepath = os.path.join(output_dir, output_file)

        create_empty_file(output_filepath)
        write_line_to_file(
            output_filepath, '\t'.join([
                'mwe', 'mwe_lemma', 'complete_mwe_in_sent', 'first_word_index',
                'first_word_orth', 'first_word_lemma', 'second_word_index',
                'second_word_orth', 'second_word_lemma', 'dir_index',
                'file_index', 'sentence'
            ]))

        xml_paths = find_xml(dir_path)

        for file_index, xml_path in enumerate(xml_paths):
            orths, lemmas = read_xml(xml_path)
            get_sentences_containing_mwe(output_filepath, mwe_list,
                                         lemmatized_mwes, orths, lemmas,
                                         restricted_words_list, dir_index,
                                         file_index)

            if file_index % 50_000 == 0 and file_index > 0:
                print(
                    f'{datetime.now().time()} - Processed {file_index} files')


if __name__ == '__main__':
    main(sys.argv[1:])
