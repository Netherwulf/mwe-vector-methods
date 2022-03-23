import sys

import numpy as np
import pandas as pd

from scipy import stats


def get_mwe_stats(filepath):
    df = pd.read_csv(filepath, sep='\t', error_bad_lines=False)

    mwe_counts = df['mwe'].value_counts().tolist()

    print(f'max: {np.max(mwe_counts)}',
          f'mode: {stats.mode(mwe_counts).mode}',
          f'mean: {np.mean(mwe_counts)}',
          f'median: {np.median(mwe_counts)}',
          f'min: {np.min(mwe_counts)}',
          sep='\n')


def main(args):
    for filepath in args:
        get_mwe_stats(filepath)


if __name__ == '__main__':
    main(sys.argv[1:])
