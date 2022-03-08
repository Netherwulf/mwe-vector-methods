import argparse
import datetime
import pickle

import numpy as np

from models.cnn import create_cnn_model


def log_message(message):
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : {message}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path',
                        help='path of the model to load',
                        type=str)
    parser.add_argument('--data_path',
                        help='path to the file containing data',
                        type=str)
    parser.add_argument('--output_path', help='path to output file', type=str)
    parser.add_argument('--column_idx',
                        help='index of column containing sample',
                        type=int)
    parser.add_argument('--skip_column_idx',
                        help='indices of columns to skip separated by colon',
                        type=str)
    parser.add_argument('--model_backend',
                        help='model backend - sk or tf',
                        type=str)
    parser.add_argument('--embedding_size',
                        help='size of the embedding vector',
                        type=str)

    args = parser.parse_args()

    return args


def load_sk_model(model_path, *kwargs):
    return pickle.load(open(model_path, 'rb'))


def get_sk_model_prediction(model, sample):

    y_pred = model.predict(sample)
    y_pred_prob = max([probs for probs in model.predict_proba(sample)])

    return y_pred, y_pred_prob


def load_tf_model(model_path, **kwargs):
    if 'input_shape' in kwargs:
        input_shape = int(kwargs['input_shape'])
    else:
        input_shape = 4 * 768

    # declare model structure
    model = create_cnn_model(input_shape=(input_shape, 1))

    # load weights from checkpoint file
    model.load_weights(model_path)

    return model


def get_tf_model_prediction(model, sample):
    sample = np.array([
        elem for elem in sample
        # for elem in np.concatenate([sample[:768 * 2], sample[768 * 3:]
        #                             ])  # for sentences containing MeWeX MWEs
        # elem for elem in np.concatenate([
        #     sample[:768 * 2], sample[768 * 5:768 * 6], sample[768 * 7:768 * 8]
        # ])  # for polish PARSEME data
    ])

    sample = np.reshape(sample, (1, sample.shape[0], 1))

    y_pred_probs = model.predict(sample)[0]

    y_pred_prob = max([probs for probs in y_pred_probs])

    y_pred = np.argmax([probs for probs in y_pred_probs])

    return y_pred, y_pred_prob


def load_model(model_backend, model_path, input_shape=(4 * 768)):
    load_dict = {'sk': load_sk_model, 'tf': load_tf_model}

    return load_dict[model_backend](model_path, input_shape=input_shape)


def get_pred(model_backend, model, sample):
    pred_dict = {'sk': get_sk_model_prediction, 'tf': get_tf_model_prediction}

    return pred_dict[model_backend](model, sample)


def get_predictions(model_path, data_path, output_path, column_idx,
                    skip_column_idx, model_backend, embedding_size):
    line_idx = 0

    skip_column_idx_list = [int(idx) for idx in skip_column_idx.split(',')]

    model = load_model(model_backend, model_path, input_shape=embedding_size)

    with open(data_path, 'r',
              buffering=2000) as in_file, open(output_path,
                                               'a',
                                               buffering=2000) as out_file:
        for line in in_file:
            line_elems = [line_elem for line_elem in line.strip().split('\t')]

            if line_idx == 0:
                column_names = [
                    column_name for i, column_name in enumerate(line_elems)
                    if i not in skip_column_idx_list
                ] + ['prediction', 'pred_prob']
                out_file.write('\t'.join(column_names) + '\n')

            else:
                sample = np.array([
                    float(elem) for elem in line_elems[column_idx].split(',')
                ])

                # check if there is an invalid embedding
                # 5 * 768 for sentences containing MeWeX MWEs
                # 2 * 768 for mean embedding
                if len(sample) != 2 * 768:
                    log_message(f'invalid sample size: {len(sample)}')
                    line_idx += 1
                    continue

                y_pred, y_pred_prob = get_pred(model_backend, model, sample)

                values_to_save = [
                    value for i, value in enumerate(line_elems)
                    if i not in skip_column_idx_list
                ] + [str(y_pred), str(y_pred_prob)]

                out_file.write('\t'.join(values_to_save) + '\n')

            line_idx += 1

            if line_idx > 0 and line_idx % 10000 == 0:
                log_message(f'Processed {line_idx} lines')


def main():
    args = parse_args()

    model_path = args.model_path
    data_path = args.data_path
    output_path = args.output_path
    column_idx = args.column_idx
    skip_column_idx = args.skip_column_idx
    model_backend = args.model_backend
    embedding_size = args.embedding_size

    get_predictions(model_path, data_path, output_path, column_idx,
                    skip_column_idx, model_backend, embedding_size)


if __name__ == '__main__':
    main()
