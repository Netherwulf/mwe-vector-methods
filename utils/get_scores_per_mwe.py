import argparse
import datetime

import numpy as np
import pandas as pd

from scipy import stats as s
from sklearn.metrics import classification_report


def log_message(message):
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : {message}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        help='path to the file containing data',
                        type=str)
    parser.add_argument('--output_path', help='path to output file', type=str)
    parser.add_argument('--extract_test',
                        action='store_true',
                        help="extract test data split")
    parser.add_argument('--save_true_labels',
                        action='store_true',
                        help="include column with true labels in output file")
    parser.add_argument('--majority_voting',
                        action='store_true',
                        help="use majority voting instead of weighted voting")

    args = parser.parse_args()

    return args


def get_majority_results(data_path, output_path, extract_test,
                         save_true_labels):
    df = pd.read_csv(data_path, sep='\t')

    if extract_test:
        df = df[df['dataset_type'] == 'test']

    mwe_weighted_pred_dict = {}

    for mwe in df['mwe'].unique().tolist():
        predictions = [
            y_pred for y_pred in df[df['mwe'] == mwe]['prediction'].tolist()
        ]

        weighted_pred = int(s.mode(predictions)[0])

        mwe_weighted_pred_dict[mwe] = weighted_pred

    df['majority_pred_is_correct'] = df['mwe'].map(mwe_weighted_pred_dict)

    mwe_df = df.drop_duplicates(subset=['mwe'])

    output_columns = ['mwe', 'majority_pred_is_correct']

    if save_true_labels:
        output_columns += ['is_correct']

        target_names = ['Incorrect MWE', 'Correct MWE']
        print(
            classification_report(mwe_df['is_correct'],
                                  mwe_df['majority_pred_is_correct'],
                                  target_names=target_names))

    mwe_df.to_csv(output_path, sep='\t', index=False, columns=output_columns)


def get_weighted_results(data_path, output_path, extract_test,
                         save_true_labels):
    df = pd.read_csv(data_path, sep='\t')

    if extract_test:
        df = df[df['dataset_type'] == 'test']

    mwe_weighted_pred_dict = {}

    for mwe in df['mwe'].unique().tolist():
        predictions_with_probs = [
            (y_pred, y_pred_prob) for y_pred, y_pred_prob in zip(
                df[df['mwe'] == mwe]['prediction'].tolist(), df[
                    df['mwe'] == mwe]['pred_prob'].tolist())
        ]

        weights_per_class = [
            0.0 for _ in range(len(df['prediction'].unique().tolist()))
        ]

        for class_id in range(len(weights_per_class)):
            weights_per_class[class_id] = sum([
                elem[1] for elem in predictions_with_probs
                if elem[0] == class_id
            ])

        weighted_pred = int(np.argmax(weights_per_class))

        mwe_weighted_pred_dict[mwe] = weighted_pred

    df['weighted_pred_is_correct'] = df['mwe'].map(mwe_weighted_pred_dict)

    mwe_df = df.drop_duplicates(subset=['mwe'])

    output_columns = ['mwe', 'weighted_pred_is_correct']

    if save_true_labels:
        output_columns += ['is_correct']

        target_names = ['Incorrect MWE', 'Correct MWE']
        print(
            classification_report(mwe_df['is_correct'],
                                  mwe_df['weighted_pred_is_correct'],
                                  target_names=target_names))

    mwe_df.to_csv(output_path, sep='\t', index=False, columns=output_columns)


def main():
    args = parse_args()

    data_path = args.data_path
    output_path = args.output_path
    extract_test = True if args.extract_test else False
    save_true_labels = True if args.save_true_labels else False

    if args.majority_voting:
        get_majority_results(data_path, output_path, extract_test,
                             save_true_labels)

    else:
        get_weighted_results(data_path, output_path, extract_test,
                             save_true_labels)


if __name__ == '__main__':
    main()
