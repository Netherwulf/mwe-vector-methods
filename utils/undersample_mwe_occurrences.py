import os
import sys

import pandas as pd


def undersample_mwe_occurrences(filepath, occurrences_num, mwe_col_ind,
                                class_col_ind, col_count, log_batch_size,
                                undersampled_class):
    filename = filepath.split('/')[-1].split('.')[0]
    dir_path = os.path.join(*filepath.split('/')[:-1])

    output_path = os.path.join(dir_path, f'{filename}_undersampled.tsv')

    mwe_dict = {}
    line_count = 0

    with open(filepath, 'r', buffering=2000000) as in_file:
        for line in in_file:

            if line_count == 0:
                with open(output_path, 'w', buffering=2000000) as out_file:
                    out_file.write(line)

                    line_count += 1

            else:
                line_attributes = line.split('\t')

                line_count += 1

                if len(line_attributes) == col_count:
                    mwe = line_attributes[mwe_col_ind]

                    if class_col_ind is not None and line_attributes[
                            class_col_ind] != undersampled_class:
                        with open(output_path, 'a') as out_file:
                            out_file.write(line)
                            continue

                    if (mwe in mwe_dict.keys()
                            and mwe_dict[mwe] == occurrences_num):
                        continue

                    if mwe not in mwe_dict.keys():
                        mwe_dict[mwe] = 1

                    else:
                        mwe_dict[mwe] += 1

                    with open(output_path, 'a') as out_file:
                        out_file.write(line)

            if line_count != 0 and line_count % log_batch_size == 0:
                print(f'Processed {line_count} lines')


def undersample_class_simple(filepath, undersampled_class):
    filename = filepath.split('/')[-1].split('.')[0]
    dir_path = os.path.join(*filepath.split('/')[:-1])
    output_path = os.path.join(dir_path, f'{filename}_undersampled.tsv')

    df = pd.read_csv(filepath, sep='\t')

    result_df = pd.DataFrame(columns=df.columns)

    for dataset_type in ['train', 'test']:
        df_not_to_undersample = df[(df['is_correct'] != undersampled_class)
                                   & (df['dataset_type'] == dataset_type)]
        df_to_undersample = df[(df['is_correct'] == undersampled_class)
                               & (df['dataset_type'] == dataset_type)]

        df_undersampled = df_to_undersample.sample(n=int(
            1.2 * len(df_not_to_undersample)),
                                                   random_state=1)

        result_df = result_df.append(df_not_to_undersample)
        result_df = result_df.append(df_undersampled)

    print(
        'Dataframe class counts:',
        f'train - 0: {len(result_df[(result_df["is_correct"] == 0) & (result_df["dataset_type"] == "train")])}',
        f'train - 1: {len(result_df[(result_df["is_correct"] == 1) & (result_df["dataset_type"] == "train")])}',
        f'test - 0: {len(result_df[(result_df["is_correct"] == 0) & (result_df["dataset_type"] == "test")])}',
        f'test - 1: {len(result_df[(result_df["is_correct"] == 1) & (result_df["dataset_type"] == "test")])}',
        sep='\n')

    result_df.to_csv(output_path, sep='\t', index=False)


def main(args):
    max_mwe_occurrences = 10
    mwe_col_ind = 7
    class_col_ind = 10
    col_count = 12
    log_batch_size = 10000
    undersampled_class = 0

    for filepath in args:
        # undersample_mwe_occurrences(filepath, max_mwe_occurrences, mwe_col_ind,
        #                             class_col_ind, col_count, log_batch_size,
        #                             undersampled_class)
        undersample_class_simple(filepath, undersampled_class)


if __name__ == '__main__':
    main(sys.argv[1:])
