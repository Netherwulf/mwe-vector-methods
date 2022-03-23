import os
import sys

import pandas as pd


def get_mwe_occurrences_samples(filepath, sample_size, samples_count):
    df = pd.read_csv(filepath, sep='\t', header=None, on_bad_lines='skip')

    df = df.rename(
        columns={
            0: 'mwe',
            1: 'is_correct',
            2: 'first_word_id',
            3: 'first_word_orth',
            4: 'first_word_lemma',
            5: 'sentence'
        })

    df = df[[
        'mwe', 'first_word_id', 'first_word_orth', 'first_word_lemma',
        'sentence'
    ]]

    start_idx = [
        sample_id * (len(df) // samples_count)
        for sample_id in range(samples_count)
    ]

    for sample_num in range(samples_count):

        filename = filepath.split('/')[-1].split('.')[0]
        dir_path = os.path.join(*filepath.split('/')[:-2])

        output_path = os.path.join(dir_path, 'preprocessed_data', 'samples',
                                   f'{filename}_sample_{sample_num}.tsv')

        subset_df = df.iloc[start_idx[sample_num]:start_idx[sample_num] +
                            (len(df) // samples_count), :]

        sample = subset_df.sample(n=sample_size, replace=False, random_state=2)

        sample = sample.drop_duplicates('mwe', ignore_index=True)

        while len(sample) != sample_size:
            temp_sample = subset_df.sample(n=sample_size - len(sample),
                                           replace=False)

            sample = sample.append(temp_sample, ignore_index=True)

            sample = sample.drop_duplicates('mwe', ignore_index=True)

        sample.to_csv(output_path, sep='\t', index=False)


def main(args):
    sample_size = 100
    samples_count = 4

    for filepath in args:
        get_mwe_occurrences_samples(filepath, sample_size, samples_count)


if __name__ == '__main__':
    main(sys.argv[1:])
