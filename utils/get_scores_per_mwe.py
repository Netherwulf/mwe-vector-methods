import argparse
import datetime
import math

import numpy as np
import pandas as pd
import spacy

from nltk import pos_tag
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
    parser.add_argument('--save_mwe_types',
                        action='store_true',
                        help="include column with MWE types in output file")
    parser.add_argument('--majority_voting',
                        action='store_true',
                        help="use majority voting instead of weighted voting")

    args = parser.parse_args()

    return args


def init_tagger(model_name='pl_core_news_lg'):
    return spacy.load(model_name)


def get_mwe_type(mwe, tagger):
    doc = tagger(mwe)

    return '+'.join([token.pos_ for token in doc])


def get_majority_results(data_path, output_path, extract_test,
                         save_true_labels, save_mwe_types, tagger):
    df = pd.read_csv(data_path, sep='\t')

    if extract_test:
        df = df[df['dataset_type'] == 'test']

    # mwe_weighted_pred_dict = {}
    # mwe_count_dict = {}
    mwe_type_dict = {}
    log_message('grouping by majority vote')
    majority_pred_df = df.groupby(
        ['mwe']).agg(majority_pred_is_correct=('prediction',
                                               pd.Series.mode)).reset_index()
    log_message('grouping by count')
    count_df = df.groupby(['mwe']).agg(count=('prediction',
                                              'count')).reset_index()

    mwe_df = df.drop_duplicates(subset=['mwe'])

    # df['majority_pred_is_correct'] = df['mwe'].map(mwe_weighted_pred_dict)
    mwe_df = mwe_df.merge(majority_pred_df[['mwe',
                                            'majority_pred_is_correct']],
                          on='mwe')

    log_message('merging with count')
    mwe_df = mwe_df.merge(count_df[['mwe', 'count']], on='mwe')
    # mwe_df['mwe_type'] = mwe_df.apply(
    #     lambda row: get_mwe_type(row['mwe'], tagger), axis=1)
    # df['count'] = df['mwe'].map(mwe_count_dict)

    output_columns = ['mwe', 'majority_pred_is_correct']

    if save_true_labels:
        output_columns += ['is_correct']

        target_names = ['Incorrect MWE', 'Correct MWE']
        print(
            classification_report(mwe_df['is_correct'],
                                  mwe_df['majority_pred_is_correct'],
                                  target_names=target_names))

    output_columns += ['count']

    if save_mwe_types:
        log_message('generating MWE types')
        for mwe in mwe_df['mwe'].tolist():
            # predictions = [
            #     y_pred for y_pred in df[df['mwe'] == mwe]['prediction'].tolist()
            # ]
            # if math.isnan(s.mode(predictions)[0]):
            #     log_message(f'predictions: {predictions}')
            # weighted_pred = int(s.mode(predictions)[0])

            # mwe_weighted_pred_dict[mwe] = weighted_pred
            # mwe_count_dict[mwe] = len(predictions)
            mwe_type_dict[mwe] = get_mwe_type(mwe, tagger)

        mwe_df['mwe_type'] = mwe_df['mwe'].map(mwe_type_dict)
        output_columns += ['mwe_type']
        log_message(f'MWE types generated')

    mwe_df.to_csv(output_path, sep='\t', index=False, columns=output_columns)


def get_weighted_results(data_path, output_path, extract_test,
                         save_true_labels, save_mwe_types, tagger):
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
    save_mwe_types = True if args.save_mwe_types else False

    tagger = init_tagger()

    if args.majority_voting:
        get_majority_results(data_path, output_path, extract_test,
                             save_true_labels, save_mwe_types, tagger)

    else:
        get_weighted_results(data_path, output_path, extract_test,
                             save_true_labels, save_mwe_types, tagger)


if __name__ == '__main__':
    main()
