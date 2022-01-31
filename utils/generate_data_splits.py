import datetime
import os
import random
import sys

import numpy as np
import pandas as pd


def assign_data_splits(filepath):
    output_dir = os.path.join(*filepath.split('/')[:-1])

    output_filename = f"{filepath.split('/')[-1].split('.')[0]}_with_splits.tsv"

    output_filepath = os.path.join(output_dir, output_filename)

    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Loading data...')
    df = pd.read_csv(filepath, sep='\t')
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Data loaded...')
    df['dataset_type'] = 'null'

    dataset_splits = ['train', 'dev', 'test']
    print(
        f'{datetime.datetime.now().strftime("%H:%M:%S")} : Setting data splits...'
    )
    for mwe in df['mwe'].unique().tolist():
        df.loc[df['mwe'] == mwe,
               'dataset_type'] = np.random.choice(dataset_splits,
                                                  size=1,
                                                  p=(0.7, 0.15, 0.15))[0]

    # mwe_split_dict = {
    #     mwe: np.random.choice(dataset_splits, size=1, p=(0.7, 0.15, 0.15))[0]
    #     for mwe in df['mwe'].unique().tolist()
    # }

    # df['data_split'] = df['mwe'].map(mwe_split_dict)
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : Saving data...')
    total_rows = len(df)
    with open(output_filepath, 'w', buffering=2000000,
              encoding="utf-8") as out_file:
        out_file.write('\t'.join(df.columns))
        out_file.write('\n')

        for ind, row in df.iterrows():
            out_file.write('\t'.join([str(elem) for elem in row]))

            if ind != total_rows - 1:
                out_file.write('\n')
    # df.to_csv(output_filepath, sep='\t', index=False)
    print(
        f'{datetime.datetime.now().strftime("%H:%M:%S")} : Saved to {output_filepath}...'
    )


def main(args):
    for filepath in args:
        assign_data_splits(filepath)


if __name__ == '__main__':
    main(sys.argv[1:])
