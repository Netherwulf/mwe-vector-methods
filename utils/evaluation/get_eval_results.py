import argparse
import datetime

import pandas as pd

from sklearn.metrics import classification_report


def log_message(message):
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : {message}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath',
                        help='path to the file containing predictions',
                        type=str)

    parser.add_argument('--class_names',
                        help='class names separated by comma',
                        type=str)

    parser.add_argument('--pred_column',
                        help='name of column containing predictions',
                        type=str)

    args = parser.parse_args()

    return args


def extract_predictions(filepath, pred_column):
    df = pd.read_csv(filepath, sep='\t')

    y_true = df['is_correct'].astype('int32')

    y_pred = df[pred_column].astype('int32')

    return y_true, y_pred


def get_evaluation_report(y_true, y_pred, target_names):

    eval_report = classification_report(y_true,
                                        y_pred,
                                        target_names=target_names,
                                        output_dict=True,
                                        digits=4)

    eval_df = pd.DataFrame(eval_report).transpose()

    print(classification_report(y_true, y_pred, target_names=target_names))

    return eval_df


def main():
    args = parse_args()

    if args.class_names:
        class_names = args.class_names.split(',')
    else:
        class_names = ['Incorrect MWE', 'Correct MWE']

    y_true, y_pred = extract_predictions(args.filepath, args.pred_column)

    get_evaluation_report(y_true, y_pred, class_names)


if __name__ == '__main__':
    main()
