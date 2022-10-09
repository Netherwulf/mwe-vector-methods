import argparse
import datetime
import os

import numpy as np
import pandas as pd
import spacy

from sklearn.metrics import classification_report


def log_message(message):
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : {message}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        help='path to the file containing data',
                        type=str)
    parser.add_argument('--save_mwe_types',
                        action='store_true',
                        help="include column with MWE types in output file")

    args = parser.parse_args()

    return args


def init_tagger(model_name='pl_core_news_lg'):
    return spacy.load(model_name)


def get_mwe_type(mwe, tagger):
    doc = tagger(mwe)

    return '+'.join([token.pos_ for token in doc])


def get_mwe_based_results(data_path, save_mwe_types, tagger):
    df = pd.read_csv(data_path, sep='\t')

    # calculating majority voting
    majority_pred_df = df.groupby(['mwe']).agg(
        majority_pred_is_correct=('prediction',
                                  lambda x: x.mode().tolist())).reset_index()

    # counting number of occurrences
    count_df = df.groupby(['mwe']).agg(count=('prediction',
                                              'count')).reset_index()

    mwe_df = df.drop_duplicates(subset=['mwe'])

    mwe_df = mwe_df.merge(majority_pred_df[['mwe',
                                            'majority_pred_is_correct']],
                          on='mwe')

    mwe_df = mwe_df.merge(count_df[['mwe', 'count']], on='mwe')

    mwe_df = mwe_df[mwe_df['majority_pred_is_correct'].str.len() == 1]

    mwe_df['majority_pred_is_correct'] = mwe_df[
        'majority_pred_is_correct'].apply(lambda x: x[0])

    weighted_pred_df = get_weighted_results(df)

    mwe_df = mwe_df.merge(weighted_pred_df[['mwe',
                                            'weighted_pred_is_correct']],
                          on='mwe')

    output_columns = [
        'mwe', 'majority_pred_is_correct', 'weighted_pred_is_correct',
        'is_correct', 'count'
    ]

    target_names = ['Incorrect MWE', 'Correct MWE']

    # majority voting evaluation results
    eval_report = classification_report(mwe_df['is_correct'],
                                        mwe_df['majority_pred_is_correct'],
                                        target_names=target_names,
                                        output_dict=True,
                                        digits=4)

    printable_eval_report = classification_report(
        mwe_df['is_correct'],
        mwe_df['majority_pred_is_correct'],
        target_names=target_names,
        digits=4)

    print('Evaluation report for majority voting:',
          printable_eval_report,
          sep='\n')

    eval_df = pd.DataFrame(eval_report).transpose().round(4)

    eval_dir = os.path.join(*data_path.split('/')[:-2], 'evaluation_results')

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    majority_eval_filepath = os.path.join(
        eval_dir,
        f'{datetime.datetime.now().date()}_{datetime.datetime.now().strftime("%H:%M:%S")}_{data_path.split("/")[-1].split(".")[0]}_evaluation_majority.tsv'
    )

    print(
        f'Saving majority-voting evaluation results to: {majority_eval_filepath}'
    )

    eval_df.to_csv(f'{majority_eval_filepath}', sep='\t')

    # weighted voting evaluation results
    eval_report = classification_report(mwe_df['is_correct'],
                                        mwe_df['weighted_pred_is_correct'],
                                        target_names=target_names,
                                        output_dict=True,
                                        digits=4)

    printable_eval_report = classification_report(
        mwe_df['is_correct'],
        mwe_df['weighted_pred_is_correct'],
        target_names=target_names,
        digits=4)

    print('Evaluation report for weighted voting:',
          printable_eval_report,
          sep='\n')

    eval_df = pd.DataFrame(eval_report).transpose().round(4)

    eval_dir = os.path.join(*data_path.split('/')[:-2], 'evaluation_results')

    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    weighted_eval_filepath = os.path.join(
        eval_dir,
        f'{datetime.datetime.now().date()}_{datetime.datetime.now().strftime("%H:%M:%S")}_{data_path.split("/")[-1].split(".")[0]}_evaluation_weighted.tsv'
    )

    print(
        f'Saving weighted-voting evaluation results to: {weighted_eval_filepath}'
    )

    eval_df.to_csv(f'{weighted_eval_filepath}', sep='\t')

    # generating MWE types column
    if save_mwe_types:
        mwe_type_dict = {}

        log_message('generating MWE types')
        for mwe in mwe_df['mwe'].tolist():
            mwe_type_dict[mwe] = get_mwe_type(mwe, tagger)

        mwe_df['mwe_type'] = mwe_df['mwe'].map(mwe_type_dict)
        output_columns += ['mwe_type']
        log_message(f'MWE types generated')

    predictions_filepath = os.path.join(
        *data_path.split('/')[:-2], 'prediction_results',
        f'{datetime.datetime.now().date()}_{datetime.datetime.now().strftime("%H:%M:%S")}_{data_path.split("/")[-1].split(".")[0]}_grouped_by_mwe.tsv'
    )

    mwe_df.to_csv(predictions_filepath,
                  sep='\t',
                  index=False,
                  columns=output_columns)


def get_weighted_results(df):
    result_df = df.copy()

    mwe_weighted_pred_dict = {}

    for mwe in result_df['mwe'].unique().tolist():
        predictions_with_probs = [
            (y_pred, y_pred_prob) for y_pred, y_pred_prob in zip(
                result_df[result_df['mwe'] == mwe]['prediction'].tolist(),
                result_df[result_df['mwe'] == mwe]['prediction_prob'].tolist())
        ]

        weights_per_class = [
            0.0 for _ in range(len(result_df['prediction'].unique().tolist()))
        ]

        for class_id in range(len(weights_per_class)):
            weights_per_class[class_id] = sum([
                elem[1] for elem in predictions_with_probs
                if elem[0] == class_id
            ])

        weighted_pred = int(np.argmax(weights_per_class))

        mwe_weighted_pred_dict[mwe] = weighted_pred

    result_df['weighted_pred_is_correct'] = result_df['mwe'].map(
        mwe_weighted_pred_dict)

    result_df = result_df.drop_duplicates(subset=['mwe'])

    return result_df


def main():
    args = parse_args()

    data_path = args.data_path
    save_mwe_types = True if args.save_mwe_types else False

    tagger = init_tagger()

    get_mwe_based_results(data_path, save_mwe_types, tagger)


if __name__ == '__main__':
    main()
