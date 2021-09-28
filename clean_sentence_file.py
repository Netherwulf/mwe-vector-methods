import sys


def clean_sentence_file(filepath):
    with open(filepath, 'r', errors='replace') as in_file:
        content = in_file.readlines()

        out_file_name = filepath.split('/')[-1].split('.')[-2] + '_cleaned.tsv'

        with open(out_file_name, 'w') as out_file:
            previous_mwe = ''
            for line in content:
                mwe = line.split('\t')[0]
                if mwe != previous_mwe:
                    out_file.write(line)
                    previous_mwe = mwe


def main(args):
    for filepath in args:
        clean_sentence_file(filepath)


if __name__ == '__main__':
    main(sys.argv[1:])
