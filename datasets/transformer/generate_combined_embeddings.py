import os
import sys


def add_combined_embeddings(filepath, complete_mwe_in_sent):
    file_name = filepath.split('/')[-1].split('.')[0]

    result_filepath = os.path.join(*filepath.split('/')[:-1],
                                   f'{file_name}_with_combined_embedding.tsv')

    with open(filepath, 'r') as in_file, open(result_filepath,
                                              'w') as out_file:
        line_idx = 0

        line = in_file.readline()

        while line:
            line = line.strip()

            if line_idx == 0:
                out_file.write(f'{line}\t"combined_embedding"\n')

                line_idx += 1

            else:
                line_attributes = line.split('\t')

                if complete_mwe_in_sent:
                    mwe_embedding = line_attributes[10]
                    first_word_only_embedding = line_attributes[11]
                    second_word_only_embedding = line_attributes[12]
                    first_word_mwe_emb_diff = line_attributes[13]
                    second_word_mwe_emb_diff = line_attributes[14]
                    first_word_mwe_emb_abs_diff = line_attributes[15]
                    second_word_mwe_emb_abs_diff = line_attributes[16]
                    first_word_mwe_emb_prod = line_attributes[17]
                    second_word_mwe_emb_prod = line_attributes[18]

                    combined_embedding = ','.join([
                        mwe_embedding, first_word_only_embedding,
                        second_word_only_embedding, first_word_mwe_emb_diff,
                        second_word_mwe_emb_diff, first_word_mwe_emb_abs_diff,
                        second_word_mwe_emb_abs_diff, first_word_mwe_emb_prod,
                        second_word_mwe_emb_prod
                    ])

                else:
                    first_word_embedding = line_attributes[10]
                    mwe_embedding = line_attributes[11]
                    first_word_mwe_emb_diff = line_attributes[12]
                    first_word_mwe_emb_abs_diff = line_attributes[13]
                    first_word_mwe_emb_prod = line_attributes[14]

                    combined_embedding = ','.join([
                        first_word_embedding, mwe_embedding,
                        first_word_mwe_emb_diff, first_word_mwe_emb_abs_diff,
                        first_word_mwe_emb_prod
                    ])

                out_file.write(f'{line}\t{combined_embedding}\n')

                line_idx += 1


def main(args):
    for filepath in args:
        print(f'\nReading file: {filepath}')

        file_name = filepath.split('/')[-1].split('.')[0]

        complete_mwe_in_sent = False if 'incomplete' in file_name.split(
            '_') else True

        print(f'\nGenerating combined embeddings for file: {filepath}')

        add_combined_embeddings(filepath, complete_mwe_in_sent)


if __name__ == '__main__':
    main(sys.argv[1:])
