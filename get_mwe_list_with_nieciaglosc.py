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


def save_mwe_to_file:



def main(args):
    for filepath in args:
        mwe_list = get_mwe_list(filepath)


if __name__ == '__main__':
    main(sys.argv[1:])
