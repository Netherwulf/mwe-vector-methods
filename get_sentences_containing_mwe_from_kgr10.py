import csv
import string
import sys
import xml.etree.ElementTree as ET
from typing import List

import morfeusz2


# read MWEs from Słowosieć tsv file
def read_mwe_from_tsv(filepath, separator, column_index) -> []:
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


def main(args):
    lemmatizer = init_lemmatizer()
    correct_mwe_list = read_mwe_from_tsv('correct_mwe.tsv', '\t', 3)
    incorrect_mwe_list = read_mwe_from_tsv('incorrect_MWE_kompozycyjne_polaczenia_plWN.csv', ',', 1)
    lemmatized_correct_mwes = lemmatize_mwe(correct_mwe_list, lemmatizer)
    lemmatized_incorrect_mwes = lemmatize_mwe(incorrect_mwe_list, lemmatizer)

    for filepath in args:
        orths, lemmas = read_xml(filepath)


if __name__ == '__main__':
    main(sys.argv[1:])
