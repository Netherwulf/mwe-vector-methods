import sys

import fasttext
import numpy as np


def load_fasttext(model_path):
    model = fasttext.load_model(model_path)

    return model


def read_mwe(file_path, incorrect_mwe_file=False):
    with open(file_path, 'r', encoding='utf-8') as correct_mwe_file:
        content = correct_mwe_file.readlines()

        mwe_list = [('*' * 200, '*' * 200) for _ in range(len(content))]

        mwe_index = 0

        for line in content:

            if incorrect_mwe_file:
                mwe = line.strip().split(',')[1]
            else:
                mwe = line.strip().split('\t')[3]

            mwe_words = mwe.split(' ')

            if len(mwe_words) != 2:
                continue

            first_word = mwe_words[0]
            second_word = mwe_words[1]

            mwe_list[mwe_index] = (first_word, second_word)

    # clean mwe_list from the dummy elements
    mwe_list = [mwe for mwe in mwe_list if mwe[0] != '*' * 200]

    return mwe_list


def generate_embeddings(ft_model, mwe_list, incorrect_mwe_list=False):
    embeddings_arr = np.empty((len(mwe_list), 901), dtype=np.float32)

    for mwe_index, mwe_words in enumerate(mwe_list):
        embeddings_arr[mwe_index][0: 300] = ft_model.get_word_vector(mwe_words[0])
        embeddings_arr[mwe_index][300: 600] = ft_model.get_word_vector(mwe_words[1])
        embeddings_arr[mwe_index][600: 900] = embeddings_arr[mwe_index][0: 300] - embeddings_arr[mwe_index][300: 600]
        embeddings_arr[mwe_index][900] = 0.0 if incorrect_mwe_list else 1.0

    return embeddings_arr


def save_mwe_embeddings(output_file_name, mwe_embeddings) -> None:
    # shuffle samples in the dataset
    np.random.shuffle(mwe_embeddings)

    # save dataset to .npy file
    np.save(output_file_name, mwe_embeddings)


def main(args):
    ft_model_path = "kgr10.plain.skipgram.dim300.neg10.bin"
    correct_mwe_file_path = 'correct_mwe.tsv'
    incorrect_mwe_file_path = 'incorrect_MWE_kompozycyjne_polaczenia_plWN.csv'
    output_file_name = 'mwe_dataset.npy'

    ft_model = load_fasttext(ft_model_path)

    embeddings_arr = np.empty((0, 901), dtype=np.float32)
    print(f'embeddings_arr basic shape: {embeddings_arr.shape}')

    for file_index, mwe_file_path in enumerate([correct_mwe_file_path, incorrect_mwe_file_path]):
        are_mwes_incorrect = file_index == 1
        mwe_list = read_mwe(mwe_file_path, incorrect_mwe_file=are_mwes_incorrect)
        mwe_embeddings = generate_embeddings(ft_model, mwe_list, incorrect_mwe_list=are_mwes_incorrect)
        print(f'mwe_embeddings shape: {mwe_embeddings.shape}')
        embeddings_arr = np.append(embeddings_arr, mwe_embeddings)

    print(f'embeddings_arr shape: {embeddings_arr.shape}')
    save_mwe_embeddings(output_file_name, embeddings_arr)


if __name__ == '__main__':
    main(sys.argv[1:])
