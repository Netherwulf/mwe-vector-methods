import os
import sys

import fasttext
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN


def load_fasttext(model_path):
    model = fasttext.load_model(model_path)

    return model


# def read_mwe(file_path, ft_model, incorrect_mwe_file):
#     df = pd.read_csv(file_path, sep='\t')
#     df = df.drop_duplicates(subset=['full_mwe'])
#     df = df[(df['first_word'].notna()) & (df['second_word'].notna()) &
#             (df['full_mwe'].notna())]

#     first_word_ft_emb_list = [
#         ft_model.get_word_vector(word) for word in df['first_word'].tolist()
#     ]
#     second_word_ft_emb = [
#         ft_model.get_word_vector(word) for word in df['second_word'].tolist()
#     ]

#     df['first_second_ft_emb_diff'] = [
#         (elem_w_1 - elem_w_2) for elem_w_1, elem_w_2 in zip(
#             first_word_ft_emb_list, second_word_ft_emb)
#     ]

#     df['first_second_ft_emb_abs_diff'] = [
#         abs(elem_w_1 - elem_w_2) for elem_w_1, elem_w_2 in zip(
#             first_word_ft_emb_list, second_word_ft_emb)
#     ]

#     df['first_second_ft_emb_prod'] = [
#         (elem_w_1 * elem_w_2) for elem_w_1, elem_w_2 in zip(
#             first_word_ft_emb_list, second_word_ft_emb)
#     ]

#     df['first_word_ft_emb'] = [
#         ','.join([str(elem) for elem in word_emb])
#         for word_emb in first_word_ft_emb_list
#     ]
#     df['second_word_ft_emb'] = [
#         ','.join([str(elem) for elem in word_emb])
#         for word_emb in second_word_ft_emb
#     ]
#     df['first_second_ft_emb_diff'] = [
#         ','.join([str(elem) for elem in word_emb])
#         for word_emb in df['first_second_ft_emb_diff'].tolist()
#     ]
#     df['first_second_ft_emb_abs_diff'] = [
#         ','.join([str(elem) for elem in word_emb])
#         for word_emb in df['first_second_ft_emb_abs_diff'].tolist()
#     ]
#     df['first_second_ft_emb_prod'] = [
#         ','.join([str(elem) for elem in word_emb])
#         for word_emb in df['first_second_ft_emb_prod'].tolist()
#     ]

#     df['is_correct'] = '0' if incorrect_mwe_file else '1'

#     return df

# with open(file_path, 'r', encoding='utf-8') as correct_mwe_file:
#     content = correct_mwe_file.readlines()
#
#     mwe_list = [('*' * 200, '*' * 200) for _ in range(len(content))]
#
#     mwe_index = 0
#
#     for line_index, line in enumerate(content):
#         if line_index == 0:
#             continue
#
#         # get MWEs from KGR10 (Słowosieć) MWE lists
#         # if incorrect_mwe_file:
#         #     if ',' in line:
#         #         mwe = line.strip().split(',')[1].replace("\"", '')
#         # else:
#         #     mwe = line.strip().split('\t')[3].replace("\"", '')
#
#         # get MWEs from PARSEME MWE lists
#         mwe = line.strip().split('\t')[7]
#
#         mwe_words = mwe.split(' ')
#
#         if len(mwe_words) != 2:
#             continue
#
#         first_word = mwe_words[0]
#         second_word = mwe_words[1]
#
#         mwe_list[mwe_index] = (first_word, second_word)
#         mwe_index += 1
#
# # clean mwe_list from the dummy elements
# mwe_list = [mwe for mwe in mwe_list if mwe[0] != '*' * 200]
#
# return mwe_list


def read_mwe(file_path, ft_model, incorrect_mwe_file):
    df = pd.read_csv(file_path, sep='\t')
    # df = df.drop_duplicates(subset=['full_mwe_lemma'])
    df = df[df['dataset_type'] == 'train'].drop_duplicates(
        subset=['full_mwe_lemma']).append(df[df['dataset_type'] != 'train'])
    df = df[(df['first_word'].notna()) & (df['second_word'].notna()) &
            (df['full_mwe'].notna())]

    first_word_ft_emb_list = [
        ft_model.get_word_vector(word) for word in df['first_word'].tolist()
    ]

    second_word_ft_emb = [
        ft_model.get_word_vector(word) for word in df['second_word'].tolist()
    ]

    df['first_second_ft_emb_diff'] = [
        (elem_w_1 - elem_w_2) for elem_w_1, elem_w_2 in zip(
            first_word_ft_emb_list, second_word_ft_emb)
    ]

    df['first_second_ft_emb_abs_diff'] = [
        abs(elem_w_1 - elem_w_2) for elem_w_1, elem_w_2 in zip(
            first_word_ft_emb_list, second_word_ft_emb)
    ]

    df['first_second_ft_emb_prod'] = [
        (elem_w_1 * elem_w_2) for elem_w_1, elem_w_2 in zip(
            first_word_ft_emb_list, second_word_ft_emb)
    ]

    combined_emb_list = np.hstack(
        (first_word_ft_emb_list, second_word_ft_emb,
         np.array(df['first_second_ft_emb_diff'].tolist()),
         np.array(df['first_second_ft_emb_abs_diff'].tolist()),
         np.array(df['first_second_ft_emb_prod'].tolist())))

    df['first_word_ft_emb'] = [
        ','.join([str(elem) for elem in word_emb])
        for word_emb in first_word_ft_emb_list
    ]

    df['second_word_ft_emb'] = [
        ','.join([str(elem) for elem in word_emb])
        for word_emb in second_word_ft_emb
    ]

    df['first_second_ft_emb_diff'] = [
        ','.join([str(elem) for elem in word_emb])
        for word_emb in df['first_second_ft_emb_diff'].tolist()
    ]

    df['first_second_ft_emb_abs_diff'] = [
        ','.join([str(elem) for elem in word_emb])
        for word_emb in df['first_second_ft_emb_abs_diff'].tolist()
    ]

    df['first_second_ft_emb_prod'] = [
        ','.join([str(elem) for elem in word_emb])
        for word_emb in df['first_second_ft_emb_prod'].tolist()
    ]

    embeddings_list = np.array([
        ','.join([str(elem) for elem in embedding])
        for embedding in combined_emb_list
    ])

    df['combined_embedding'] = embeddings_list

    return df


def get_count_dict(file_path, incorrect_mwe_file=False, type_based=False):
    with open(file_path, 'r', encoding='utf-8',
              buffering=2000000) as correct_mwe_file:
        content = correct_mwe_file.readlines()

        count_dict = {}

        for line_index, line in enumerate(content):
            if line_index == 0:
                continue

            if incorrect_mwe_file:
                if ',' not in line or len(line.split(',')) < 4:
                    continue

                if type_based:
                    domain = line.strip().split(',')[3]
                else:
                    print(f'Line: {line}')
                    domain = line.strip().split(',')[2]
            else:
                if type_based:
                    domain = line.strip().split('\t')[1]
                else:
                    domain = line.strip().split('\t')[2]

            if domain in count_dict.keys():
                count_dict[domain] += 1

            else:
                count_dict[domain] = 1

    return count_dict


def read_mwe_balanced(file_path,
                      count_dict,
                      final_sample_size=4064,
                      incorrect_mwe_file=False,
                      type_based=False):
    imbalanced_sample_size = 0

    for domain in count_dict.keys():
        imbalanced_sample_size += count_dict[domain]

    with open(file_path, 'r', encoding='utf-8',
              buffering=2000000) as correct_mwe_file:
        content = correct_mwe_file.readlines()

        mwe_list = [('*' * 200, '*' * 200) for _ in range(len(content))]

        mwe_index = 0

        mwe_domain_dict = {}

        for line_index, line in enumerate(content):
            if line_index == 0:
                continue

            if incorrect_mwe_file:
                if ',' not in line or len(line.split(',')) < 4:
                    continue

                mwe = line.strip().split(',')[1].replace("\"", '')

                if type_based:
                    domain = line.strip().split(',')[3]
                else:
                    domain = line.strip().split(',')[2]

            else:
                mwe = line.strip().split('\t')[3].replace("\"", '')

                if type_based:
                    domain = line.strip().split('\t')[1]
                else:
                    domain = line.strip().split('\t')[2]

            if domain in mwe_domain_dict.keys() \
                    and mwe_domain_dict[domain] >= int(
                (count_dict[domain] / imbalanced_sample_size) * final_sample_size):
                continue

            mwe_words = mwe.split(' ')

            if len(mwe_words) != 2:
                continue

            first_word = mwe_words[0]
            second_word = mwe_words[1]

            mwe_list[mwe_index] = (first_word, second_word)
            mwe_index += 1

            if domain in mwe_domain_dict.keys():
                mwe_domain_dict[domain] += 1
            else:
                mwe_domain_dict[domain] = 1

    # clean mwe_list from the dummy elements
    mwe_list = [mwe for mwe in mwe_list if mwe[0] != '*' * 200]

    return mwe_list


def generate_embeddings(ft_model, mwe_list, incorrect_mwe_list=False):
    embeddings_arr = np.empty((len(mwe_list), 901), dtype=np.float32)

    for mwe_index, mwe_words in enumerate(mwe_list):
        embeddings_arr[mwe_index][0:300] = ft_model.get_word_vector(
            mwe_words[0])
        embeddings_arr[mwe_index][300:600] = ft_model.get_word_vector(
            mwe_words[1])
        embeddings_arr[mwe_index][600:900] = embeddings_arr[mwe_index][
            0:300] - embeddings_arr[mwe_index][300:600]
        embeddings_arr[mwe_index][900] = 0.0 if incorrect_mwe_list else 1.0

    return embeddings_arr


def save_mwe_embeddings(output_file_name, mwe_embeddings) -> None:
    # shuffle samples in the dataset
    np.random.shuffle(mwe_embeddings)

    # save dataset to .npy file
    np.save(output_file_name, mwe_embeddings)


def get_smote_oversampler(smote_key):
    smote_dict = {
        'smote': SMOTE(),
        'borderline': BorderlineSMOTE(),
        'svm': SVMSMOTE(),
        'adasyn': ADASYN()
    }

    return smote_dict[smote_key]


def main(args):
    # ft_model_path = "kgr10.plain.skipgram.dim300.neg10.bin"
    ft_model_path = os.path.join('storage', 'pretrained_models',
                                 'kgr10.plain.cbow.dim300.neg10.bin')
    # change filepaths to correct and incorrect MWE lists depending on the dataset
    # Słowosieć (KGR10) correct and incorrect MWEs
    # correct_mwe_file_path = 'correct_mwe.tsv'
    # incorrect_mwe_file_path = 'incorrect_MWE_kompozycyjne_polaczenia_plWN.csv'
    # change output file name depending on the domain/type balance strategy
    # output_file_name = 'mwe_dataset_type_balanced.npy'
    # output_file_name = 'mwe_dataset_cbow.npy'

    # PARSEME Polish correct and incorrect MWEs
    # correct_mwe_file_path = 'parseme_correct_mwes.tsv'
    # incorrect_mwe_file_path = 'parseme_incorrect_mwes.tsv'
    input_filepath = os.path.join('storage', 'parseme', 'pl',
                                  'preprocessed_data', 'parseme_data.tsv')

    # result_dir_name = os.path.join('storage', 'parseme', 'pl', 'embeddings',
    #                                'fasttext')

    result_dir_name = os.path.join('storage', 'parseme', 'pl', 'embeddings',
                                   'fasttext_dupplicates')

    if not os.path.exists(result_dir_name):
        os.mkdir(result_dir_name)

    output_file_name = os.path.join(result_dir_name,
                                    'parseme_pl_embeddings.tsv')

    ft_model = load_fasttext(ft_model_path)

    # embeddings_arr = np.empty((0, 901), dtype=np.float32)

    # correct_mwe_df = read_mwe(correct_mwe_file_path, ft_model, False)
    # incorrect_mwe_df = read_mwe(incorrect_mwe_file_path, ft_model, True)

    # mwe_df = correct_mwe_df.append(incorrect_mwe_df)

    df = read_mwe(input_filepath, ft_model, False)

    df.to_csv(output_file_name, index=False, sep='\t')

    # SMOTE, Borderline SMOTE, SVM-SMOTE and ADASYN dataset variants
    for smote_type in ['smote', 'borderline', 'svm', 'adasyn']:
        print(f'Generating {smote_type} dataset variant...')
        oversample = get_smote_oversampler(smote_type)

        X_train = df[df['dataset_type'] ==
                     'train']['combined_embedding'].tolist()

        X_train = np.array([
            np.array([float(elem) for elem in embedding.split(',')])
            for embedding in X_train
        ])

        y_train = df[df['dataset_type'] == 'train']['is_correct'].tolist()

        transformed_X_train, transformed_y_train = oversample.fit_resample(
            X_train, y_train)

        transformed_X_train = np.array([
            ','.join([str(elem) for elem in embedding])
            for embedding in transformed_X_train
        ])

        transformed_y_train = np.array(
            [str(label) for label in transformed_y_train])

        transformed_df = pd.DataFrame({
            'combined_embedding': transformed_X_train,
            'is_correct': transformed_y_train
        })

        transformed_df.to_csv(os.path.join(
            result_dir_name, f'parseme_pl_embeddings_train_{smote_type}.tsv'),
                              sep='\t',
                              index=False)

    # for file_index, mwe_file_path in enumerate([correct_mwe_file_path, incorrect_mwe_file_path]):
    #     are_mwes_incorrect = file_index == 1
    #
    #     # imbalanced mwe dataset generation
    #     mwe_list = read_mwe(mwe_file_path, incorrect_mwe_file=are_mwes_incorrect)

    # domain balanced or type balanced mwe dataset generation
    # count_dict = get_count_dict(mwe_file_path, incorrect_mwe_file=are_mwes_incorrect, type_based=True)
    # mwe_list = read_mwe_balanced(mwe_file_path, count_dict, final_sample_size=4064,
    #                              incorrect_mwe_file=are_mwes_incorrect, type_based=True)

    # mwe_embeddings = generate_embeddings(ft_model, mwe_list, incorrect_mwe_list=are_mwes_incorrect)
    #
    # embeddings_arr = np.concatenate((embeddings_arr, mwe_embeddings), axis=0)

    # save_mwe_embeddings(output_file_name, embeddings_arr)


if __name__ == '__main__':
    main(sys.argv[1:])
