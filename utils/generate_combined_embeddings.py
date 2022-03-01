import argparse
import datetime
import os

import fasttext
import numpy as np


def log_message(message):
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : {message}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_filepath',
                        help='path to the file containing data',
                        type=str)
    parser.add_argument('--out_filepath',
                        help='path to the file containing data',
                        type=str)
    parser.add_argument(
        '--embedding_types',
        help='additional combined embedding type names separated by comma',
        type=str)

    args = parser.parse_args()

    return args


def load_fasttext(model_path):
    model = fasttext.load_model(model_path)

    return model


def get_columns_dict(line_elems):
    return {line_elems[ind]: ind for ind in range(len(line_elems))}


def get_baseline_emb(line_elems, column_dict):
    column_names = [
        'first_word_only_embedding', 'mwe_embedding',
        'first_word_mwe_emb_abs_diff', 'first_word_mwe_emb_prod'
    ]

    cols_idx = [column_dict[col_name] for col_name in column_names]

    return ','.join([line_elems[col_idx] for col_idx in cols_idx])


def get_ft_emb(line_elems, column_dict, ft_model):
    return [
        ','.join([
            str(elem) for elem in ft_model.get_word_vector(line_elems[
                column_dict['first_word']])
        ]), ','.join([
            str(elem) for elem in ft_model.get_word_vector(line_elems[
                column_dict['second_word']])
        ])
    ]


def get_diff_emb(line_elems, column_dict):
    # get difference between component embeddings`
    first_word_emb = np.array([
        float(elem)
        for elem in line_elems[column_dict['first_word_ft_emb']].split(',')
    ])
    second_word_emb = np.array([
        float(elem)
        for elem in line_elems[column_dict['second_word_ft_emb']].split(',')
    ])

    # get element-wise difference between above vectors
    component_diff = first_word_emb - second_word_emb

    # get difference between component-MWE diff vectors
    first_comp_mwe_diff = np.array([
        float(elem) for elem in line_elems[
            column_dict['first_word_mwe_emb_abs_diff']].split(',')
    ])
    second_comp_mwe_diff = np.array([
        float(elem) for elem in line_elems[
            column_dict['second_word_mwe_emb_abs_diff']].split(',')
    ])

    # get element-wise absolute between above vectors
    avg_diff = np.mean([first_comp_mwe_diff, second_comp_mwe_diff], axis=0)

    # concat component_diff and avg_diff
    diff_emb = np.concatenate([component_diff, avg_diff], axis=0)

    return ','.join([str(elem) for elem in diff_emb])


def get_prod_emb(line_elems, column_dict):
    # get component embeddings`
    first_word_emb = np.array([
        float(elem)
        for elem in line_elems[column_dict['first_word_ft_emb']].split(',')
    ])
    second_word_emb = np.array([
        float(elem)
        for elem in line_elems[column_dict['second_word_ft_emb']].split(',')
    ])

    # get element-wise Hadamard product between above vectors
    component_prod = first_word_emb * second_word_emb

    # get Hadamard products between componenta and MWE
    first_comp_mwe_prod = np.array([
        float(elem) for elem in line_elems[
            column_dict['first_word_mwe_emb_prod']].split(',')
    ])
    second_comp_mwe_prod = np.array([
        float(elem) for elem in line_elems[
            column_dict['second_word_mwe_emb_prod']].split(',')
    ])

    # get element-wise average between above vectors
    avg_prod = np.mean([first_comp_mwe_prod, second_comp_mwe_prod], axis=0)

    # concat component_diff and avg_diff
    prod_emb = np.concatenate([component_prod, avg_prod], axis=0)

    return ','.join([str(elem) for elem in prod_emb])


def get_mean_emb(line_elems, column_dict):
    # get difference between component-MWE diff vectors
    first_comp_mwe_diff = np.array([
        float(elem) for elem in line_elems[
            column_dict['first_word_mwe_emb_abs_diff']].split(',')
    ])
    second_comp_mwe_diff = np.array([
        float(elem) for elem in line_elems[
            column_dict['second_word_mwe_emb_abs_diff']].split(',')
    ])

    # get element-wise average between above vectors
    avg_diff = np.mean([first_comp_mwe_diff, second_comp_mwe_diff], axis=0)

    # get mean Hadamard product between component and the MWE
    first_comp_mwe_prod = np.array([
        float(elem) for elem in line_elems[
            column_dict['first_word_mwe_emb_prod']].split(',')
    ])
    second_comp_mwe_prod = np.array([
        float(elem) for elem in line_elems[
            column_dict['second_word_mwe_emb_prod']].split(',')
    ])

    # get element-wise average between above vectors
    avg_prod = np.mean([first_comp_mwe_prod, second_comp_mwe_prod], axis=0)

    # concat avg_diff and avg_prod
    diff_emb = np.concatenate([avg_diff, avg_prod], axis=0)

    return ','.join([str(elem) for elem in diff_emb])


def get_emb(line_elems, emb_type, columns_dict):
    emb_dict = {
        'baseline': get_baseline_emb,
        'diff_emb': get_diff_emb,
        'prod_emb': get_prod_emb,
        'mean_emb': get_mean_emb
    }

    return emb_dict[emb_type](line_elems, columns_dict)


def generate_embedding(filepath, out_filepath, embedding_types, ft_model_path):
    ft_model = load_fasttext(ft_model_path)
    base_output_filename = out_filepath.split('/')[-1].split('.')[0]

    for emb_type in embedding_types:
        log_message(f'Generating {emb_type} embeddings')

        out_filename = f"{base_output_filename}_{emb_type}.tsv"

        out_filepath = '/'.join(out_filepath.split('/')[:-1] + [out_filename])

        with open(filepath, 'r', encoding='utf-8',
                  buffering=2000) as in_file, open(out_filepath,
                                                   'a',
                                                   encoding='utf-8',
                                                   buffering=2000) as out_file:
            line_idx = 0

            for line in in_file:
                if line_idx == 0:
                    column_names = line.rstrip().split('\t')

                    emb_column_names = [
                        column_name for column_name in column_names
                        if 'emb' in column_name
                    ]
                    emb_column_idx = [
                        column_names.index(column_name)
                        for column_name in emb_column_names
                    ]

                    column_names += ['first_word_ft_emb', 'second_word_ft_emb']

                    column_names += [emb_type]

                    columns_dict = get_columns_dict(column_names)

                    column_names = [
                        column_name
                        for i, column_name in enumerate(column_names)
                        if i not in emb_column_idx
                    ]

                    out_file.write(
                        '\t'.join([column for column in column_names]) + '\n')

                    line_idx += 1
                    continue

                else:
                    line_elems = line.rstrip().split('\t')

                    line_elems += get_ft_emb(line_elems, columns_dict,
                                             ft_model)

                    line_elems += [get_emb(line_elems, emb_type, columns_dict)]

                    line_elems = [
                        elem for i, elem in enumerate(line_elems)
                        if i not in emb_column_idx
                    ]

                    out_file.write(
                        '\t'.join([line_elem
                                   for line_elem in line_elems]) + '\n')

                    line_idx += 1

                if line_idx != 0 and line_idx % 10000 == 0:
                    log_message(f'Processed {line_idx} files...')


def main():
    args = parse_args()

    in_filepath = args.in_filepath
    out_filepath = args.out_filepath
    embedding_types = args.embedding_types.split(',')

    ft_model_path = os.path.join('storage', 'pretrained_models',
                                 'cc.en.300.bin')

    generate_embedding(in_filepath, out_filepath, embedding_types,
                       ft_model_path)


if __name__ == '__main__':
    main()
