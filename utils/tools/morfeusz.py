import string

import morfeusz2

from typing import List


# init Morfeusz2 lemmatizer
def init_lemmatizer():
    return morfeusz2.Morfeusz()  # initialize Morfeusz object


# lemmatize single MWE
def lemmatize_single_mwe(mwe, lemmatizer) -> str:

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
        mwe_words = [token for token in mwe.split(' ')]
        lemmatized_mwe_list[i] = ' '.join([
            str(lemmatizer.analyse(word)[0][2][1])
            if word not in string.punctuation else word for word in mwe_words
        ])

    return lemmatized_mwe_list
