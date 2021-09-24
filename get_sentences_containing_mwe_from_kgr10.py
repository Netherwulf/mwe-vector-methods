import csv
import string
import sys
import xml.etree.ElementTree as ET

from glob import glob
from typing import List

import morfeusz2


# find morph.xml.ccl.xml files in dir recursively
def find_xml(dir_path) -> List[str]:
    return [filepath for filepath in glob(dir_path + '/**/*.ccl.xml', recursive=True)]


# read MWEs from Słowosieć tsv or csv file
def read_mwe(filepath, separator, column_index) -> []:
    with open(filepath, 'r', encoding="utf-8") as f:
        content = list(csv.reader(f, delimiter=separator, quotechar='"'))
        mwe_list = [sublist[column_index] for sublist in content[1:] if len(sublist) != 0]

        return mwe_list


# read plain text from XML file
def read_xml(filepath) -> (List, List):
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
        lemmatized_mwe_list[i] = ' '.join(
            [str(lemmatizer.analyse(word)[0][2][1]) if word not in string.punctuation else word for word in mwe_words])

    return lemmatized_mwe_list


def load_mwes(correct_mwes_file, incorrect_mwes_file):
    lemmatizer = init_lemmatizer()

    correct_mwe_list = read_mwe(correct_mwes_file, '\t', 3)
    incorrect_mwe_list = read_mwe(incorrect_mwes_file, ',', 1)
    lemmatized_correct_mwes = lemmatize_mwe(correct_mwe_list, lemmatizer)
    lemmatized_incorrect_mwes = lemmatize_mwe(incorrect_mwe_list, lemmatizer)

    return correct_mwe_list, incorrect_mwe_list, lemmatized_correct_mwes, lemmatized_incorrect_mwes


def create_empty_file(filepath):
    with open(filepath, 'w') as f:
        pass


def get_sentences_containing_mwe(output_file, correct_mwes, incorrect_mwes, lemmatized_correct_mwes,
                                 lemmatized_incorrect_mwes, sentences_orths, sentences_lemmas):
    for sentence_ind, sentence in enumerate(sentences_orths):
        for orth_ind, orth in enumerate(sentence):


def main(args):
    correct_mwes_filepath = 'correct_mwe.tsv'
    incorrect_mwes_filepath = 'incorrect_MWE_kompozycyjne_polaczenia_plWN.csv'
    output_file = 'sentences_containing_mwe_from_kgr10.tsv'

    create_empty_file(output_file)

    correct_mwe_list, incorrect_mwe_list, lemmatized_correct_mwes, lemmatized_incorrect_mwes = load_mwes(
        correct_mwes_filepath, incorrect_mwes_filepath)

    for dir_path in args:
        xml_paths = find_xml(dir_path)

        for xml_path in xml_paths:
            orths, lemmas = read_xml(xml_path)
            get_sentences_containing_mwe(output_file, correct_mwe_list, incorrect_mwe_list,
                                         lemmatized_correct_mwes, lemmatized_incorrect_mwes, orths, lemmas)


if __name__ == '__main__':
    main(sys.argv[1:])
