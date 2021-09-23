import csv
import datetime
import operator
import os
import string
import xml.etree.ElementTree as ET

from glob import glob
from shutil import copyfile
from stempel import StempelStemmer
from typing import List

import numpy as np


def init_lemmatizer() -> StempelStemmer:
    return StempelStemmer.polimorf()


# read MWEs from Słowosieć tsv file
def read_mwe(filepath) -> []:
    with open(filepath, 'r', encoding="utf-8") as f:
        content = list(csv.reader(f, delimiter='\t'))
        mwe_list = [sublist[2] for sublist in content[1:]]

        return mwe_list


# read text from file - UNUSED IN PRODUCTION
def read_text(filepath) -> str:
    with open(filepath, 'r', encoding="utf-8") as f:
        content = f.readlines()
        content = map(lambda line: line.strip(), content)

        return ' '.join(content)


# read plain text from XML file
def read_xml(filepath) -> str:
    tree = ET.parse(filepath)
    words = [node.text for node in tree.iter('base')]

    return ' '.join(words)


# remove punctuation and lemmatize words
def lemmatize_words(text, lemmatizer) -> List[str]:
    words = [token.lower() for token in text.split(' ') if token not in string.punctuation]
    lemmatized_words = [lemmatizer.stem(word) for word in words]

    return lemmatized_words


# lemmatize MWEs
def lemmatize_mwe(mwe_list, lemmatizer) -> List[str]:
    lemmatized_mwe_list = ['*' * 200 for _ in range(len(mwe_list))]

    for i, mwe in enumerate(mwe_list):
        mwe_words = [token.lower() for token in mwe.split(' ') if token not in string.punctuation]
        lemmatized_mwe_list[i] = ' '.join([str(lemmatizer.stem(word)) for word in mwe_words])

    return lemmatized_mwe_list


# check if text contains specified MWE
def check_mwe_occurence(mwe_words, lemmatized_words) -> bool:
    for mwe_word in mwe_words:
        if mwe_word not in lemmatized_words:
            return False
        else:
            lemmatized_words.remove(mwe_word)

    return True


# count MWE in text
def count_mwe(lemmatized_mwe_list, lemmatized_words) -> (int, List[str], str):
    mwe_count = 0

    found_mwe = ['*' * 200 for _ in range(len(lemmatized_mwe_list))]

    lemmatized_text = [str(word) for word in lemmatized_words]

    for ind, mwe in enumerate(lemmatized_mwe_list):
        mwe_words = mwe.split(' ')

        # check if MWE occurs in text
        # while all(mwe_word in lemmatized_words for mwe_word in mwe_words):
        while check_mwe_occurence(mwe_words, lemmatized_words):
            # increase MWE count
            mwe_count += 1
            found_mwe[mwe_count] = mwe

            # remove MWE words from text
            # for mwe_word in mwe_words:
            #     lemmatized_words.remove(mwe_word)

    found_mwe = [mwe for mwe in found_mwe if mwe != '*' * 200]

    return mwe_count, found_mwe, ' '.join(lemmatized_text)


# find morph.xml.ccl.xml files in dir recursively
def find_xml(dir_path) -> List[str]:
    return [filepath for filepath in glob(dir_path + '/**/*.ccl.xml', recursive=True)]


# save found MWE to csv file
def save_found_mwe(filepath, found_mwe) -> None:
    dir_path = '/' + '/'.join(filepath.split('/')[:-1])
    file_name = filepath.split('/')[-1].split('.')[0]

    with open(os.path.join(dir_path, file_name + '_mwe.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(found_mwe)


# get dict (key = filepath : value = number of MWEs)
def get_file_dict(dir_path, lemmatized_mwe_list, lemmatizer_instance) -> {str: int}:
    xml_paths = find_xml(dir_path)
    file_dict = {}

    for i, xml_path in enumerate(xml_paths):
        text = read_xml(xml_path)
        lemmatized_text = lemmatize_words(text, lemmatizer_instance)
        mwe_count, found_mwe, text = count_mwe(lemmatized_mwe_list, lemmatized_text)

        file_dict[xml_path] = mwe_count

        save_found_mwe(xml_path, found_mwe)

        if i % 10000 == 0:
            print(f'{datetime.datetime.now().strftime("%H:%M:%S")} :',
                  f'\nProcessed {i} / {len(xml_paths)}',
                  f'\nCurrent file MWE count: {mwe_count}')

    return file_dict


def main():
    # path to tsv containing MWEs
    mwe_path = '/data4/netherwulf/kgr10_train_test_split/mwe.tsv'

    # root dir for all dirs
    root_dir = '/data4/netherwulf/kgr10_train_test_split/kgr10_splitted_files'

    # path to dir containing ccl.xml files
    dirs_list = ['group_0', 'group_1', 'group_2', 'group_3', 'group_4']

    # initialize lemmatizer
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Initializing lemmatizer...')
    stempel = init_lemmatizer()

    # read MWE list from file
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Reading MWE list from file...')
    mwe_list = read_mwe(mwe_path)

    # lemmatize MWEs
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Lemmatizing MWEs...')
    lemmatized_mwe_list = lemmatize_mwe(mwe_list, stempel)

    # get file dict (key = filepath : value = number of MWEs)
    for dir_name in dirs_list:
        print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Generating MWE files for dir {dir_name}...')
        get_file_dict(os.path.join(root_dir, dir_name), lemmatized_mwe_list, stempel)


if __name__ == '__main__':
    main()
