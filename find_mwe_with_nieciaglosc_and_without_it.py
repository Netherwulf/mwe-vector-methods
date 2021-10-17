import sys

import numpy as np


def get_mwes_with_nieciaglosc_and_without_it(filepath):
    with open(filepath, 'r') as file:
        print(f'Reading content of file: {filepath}')
        content = file.readlines()
        print('Initializing mwe_with_nieciaglosc list')
        mwe_with_nieciaglosc = np.array(['*' * 200 for _ in range(int(len(content)))])
        print('Initializing mwe_without_nieciaglosc list')
        mwe_without_nieciaglosc = np.array(['*' * 200 for _ in range(int(len(content)))])
        print('Initializing mwe_with_and_without_nieciaglosc list')
        mwe_with_and_without_nieciaglosc = np.array([('*' * 20, '*' * 50, '*' * 200) for _ in range(int(len(content)))])
        print('Finished initializing lists...')
        mwe_with_nieciaglosc_count = 0
        mwe_without_nieciaglosc_count = 0
        mwe_with_and_without_nieciaglosc_count = 0

        for i, line in enumerate(content):
            line = line.strip('\n')
            measure_value = line.split('\t')[0]
            mwe_types = line.split('\t')[1].split(' + ')
            mwe = line.split('\t')[2]

            if 'nieciągłość' in mwe_types:

                if mwe not in mwe_with_nieciaglosc:
                    mwe_with_nieciaglosc[mwe_with_nieciaglosc_count] = mwe
                    mwe_with_nieciaglosc_count += 1

                if mwe in mwe_without_nieciaglosc:
                    print(f'New mwe with and without nieciągłość: {mwe}')
                    mwe_with_and_without_nieciaglosc[mwe_with_and_without_nieciaglosc_count] = (measure_value, ' + '.join(mwe_types), mwe)
                    mwe_with_and_without_nieciaglosc_count += 1

            if 'nieciągłość' not in mwe_types:

                if mwe not in mwe_without_nieciaglosc:
                    mwe_without_nieciaglosc[mwe_without_nieciaglosc_count] = mwe
                    mwe_without_nieciaglosc_count += 1

                if mwe in mwe_with_nieciaglosc:
                    print(f'New mwe with and without nieciągłość: {mwe}')
                    mwe_with_and_without_nieciaglosc[mwe_with_and_without_nieciaglosc_count] = (measure_value, ' + '.join(mwe_types), mwe)
                    mwe_with_and_without_nieciaglosc_count += 1

            if i % 1000000 == 0 and i > 0:
                print(f'Processed {i} files...')

        mwe_with_and_without_nieciaglosc = np.array(elem for elem in mwe_with_and_without_nieciaglosc if elem != ('*' * 20, '*' * 50, '*' * 200))

        return mwe_with_and_without_nieciaglosc


def save_mwes_to_tsv(mwe_tuples_list):
    with open('mwe_with_and_without_nieciaglosc.tsv', 'r') as out_file:
        for mwe_tuple in mwe_tuples_list:
            out_file.write(f'{mwe_tuple[0]}\t{mwe_tuple[1]}\t{mwe_tuple[2]}\n')


def main(args):
    for filepath in args:
        mwe_typle_list = get_mwes_with_nieciaglosc_and_without_it(filepath)
        save_mwes_to_tsv(mwe_typle_list)


if __name__ == '__main__':
    main(sys.argv[1:])
