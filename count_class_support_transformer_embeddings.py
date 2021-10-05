import sys

import pandas as pd


def count_support(filepath):
    df = pd.read_csv(filepath, sep='\t', header=None, usecols=[3])

    incorrect_count = df[0].tolist().count(0)
    correct_count = df[0].tolist().count(1)

    print(f'incorrect samples: {incorrect_count}',
          f'correct samples: {correct_count}',
          f'incorrect / correct: {round(incorrect_count / correct_count, 2)}',
          f'correct / incorrect: {round(correct_count / incorrect_count, 2)}',
          f'correct / all: {round(correct_count / len(df[0]), 2)}',
          f'incorrect / all: {round(incorrect_count / len(df[0]), 2)}',
          sep='\n')


def main(args):
    for filepath in args:
        count_support(filepath)


if __name__ == '__main__':
    main(sys.argv[1:])
