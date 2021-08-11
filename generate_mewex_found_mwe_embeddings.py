from generate_fasttext_embeddings import save_mwe_embeddings, generate_embeddings
from logistic_regression import load_lr_model, get_lr_model_predictions

import os
import sys

import fasttext
import numpy as np
import pandas as pd


def load_fasttext(model_path):
    model = fasttext.load_model(model_path)

    return model


def read_mewex_found_mwe(filepath):
    df = pd.read_csv(filepath, header=None, sep='\t')

    mwe_list = np.array([('*' * 200, '*' * 200) for _ in range(len(df))])

    for i, mwe in enumerate(df.iloc[:, 2].tolist()):
        mwe_words = mwe.split(' ')
        mwe_list[i] = (mwe_words[0], mwe_words[1])

    return df, mwe_list


def main(args):
    ft_model_path = "kgr10.plain.skipgram.dim300.neg10.bin"
    lr_model_path = os.path.join('models', 'lr_model.pkl')
    mewex_found_mwe_filepath = os.path.join('/data4', 'netherwulf', 'kgr10_train_test_split',
                                            'scaled_vector_association_measure_correct_mwe_best_f1.tsv')
    print('Loading fasttext model...')
    ft_model = load_fasttext(ft_model_path)
    print(f'Reading MWEs from file: {mewex_found_mwe_filepath}')
    df, mwe_list = read_mewex_found_mwe(mewex_found_mwe_filepath)
    print('Generating fasttext embeddings...')
    embeddings_arr = generate_embeddings(ft_model, mwe_list)

    embeddings_arr = np.array([elem[:-1] for elem in embeddings_arr])
    print('Loading LR model')
    lr_model = load_lr_model(lr_model_path)
    print('Generating LR model predictions')
    y_pred = get_lr_model_predictions(lr_model, embeddings_arr)

    df['is_correct'] = y_pred
    print('Saving final TSV file...')
    df.to_csv('scaled_vector_association_measure_correct_mwe_best_f1_with_lr_prediction.tsv', sep='\t')


if __name__ == '__main__':
    main(sys.argv[1:])
