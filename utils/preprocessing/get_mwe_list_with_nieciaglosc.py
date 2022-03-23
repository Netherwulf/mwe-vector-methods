import sys


def get_mwe_list(filepath):
    with open(filepath, 'r', errors='replace') as in_file:
        content = in_file.readlines()

        mwe_list = []

        for i, line in content, enumerate():
            mwe = ' '.join(line.strip().split(' ')[-2:])

            if mwe not in mwe_list:
                mwe_list.append(mwe)

        return mwe_list


def save_mwe_to_file(mwe_list, output_filepath):
    with open(output_filepath, 'w') as out_file:
        for mwe in mwe_list:
            out_file.write(f'{mwe}\n')


def main(args):
    for filepath in args:
        mwe_list = get_mwe_list(filepath)
        output_filepath = filepath.split('.') + 'found_mwe.csv'

        save_mwe_to_file(mwe_list, output_filepath)


if __name__ == '__main__':
    main(sys.argv[1:])
