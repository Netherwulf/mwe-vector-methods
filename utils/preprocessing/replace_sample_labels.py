import argparse
import datetime

from utils.preprocessing.statistics.count_mwe_occurrences import read_mwe, init_lemmatizer, lemmatize_single_mwe


def log_message(message):
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} : {message}')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        help='path to the file containing sentences',
                        type=str)
    parser.add_argument('--data_mwe_idx',
                        help='index of column containing MWE in data file',
                        type=int)
    parser.add_argument('--label_idx',
                        help='index of column containing labels',
                        type=int)
    parser.add_argument('--mwe_list_path', help='path to MWE list', type=str)
    parser.add_argument('--output_path',
                        help='path to the output file',
                        type=str)
    parser.add_argument('--mwe_col_idx',
                        help='index of column containing MWE in MWE list',
                        type=int)

    args = parser.parse_args()

    return args


def replace_sample_labels(data_path,
                          data_mwe_idx,
                          label_idx,
                          mwe_list_path,
                          mwe_col_idx,
                          output_path,
                          separator='\t'):
    lemmatizer = init_lemmatizer()

    mwe_list = read_mwe(mwe_list_path, mwe_col_idx, lemmatizer)

    with open(data_path, 'r') as in_file, open(output_path, 'w') as out_file:
        line_idx = 0

        for line in in_file:
            line_elems = line.strip().split(separator)

            if line_idx == 0:
                line_elems = line_elems[1:]

            else:
                mwe = line_elems[data_mwe_idx]

                lemmatized_mwe = lemmatize_single_mwe(mwe, lemmatizer)

                if lemmatized_mwe in mwe_list:
                    line_elems[label_idx] = '0'

            out_file.write('\t'.join(line_elems) + '\n')

            line_idx += 1

            if line_idx != 0 and line_idx % 10000 == 0:
                log_message(f'Processed {line_idx} lines')


def main():
    args = parse_args()

    data_path = args.data_path
    data_mwe_idx = args.data_mwe_idx
    label_idx = args.label_idx
    mwe_list_path = args.mwe_list_path
    mwe_col_idx = args.mwe_col_idx
    output_path = args.output_path

    replace_sample_labels(data_path, data_mwe_idx, label_idx, mwe_list_path,
                          mwe_col_idx, output_path)


if __name__ == '__main__':
    main()
