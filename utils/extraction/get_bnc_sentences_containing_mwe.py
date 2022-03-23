import datetime
import os
import string

import nltk
import pandas as pd

from more_itertools import locate
from nltk import BigramCollocationFinder, word_tokenize, pos_tag
from nltk.corpus.reader import BNCCorpusReader
from nltk.collocations import BigramAssocMeasures


def get_curr_time():
    return f'{datetime.datetime.now().strftime("%H:%M:%S")}'


def save_sentence_df(sentence_df, filepath):
    with open(filepath, 'w', encoding='utf8') as f_out:
        for row_idx in range(len(sentence_df)):
            f_out.write('\t'.join(sentence_df.loc[row_idx, :].values.tolist()))


def write_row_to_file(filepath, row, append=False):
    mode = 'a' if append else 'w'
    with open(filepath, mode, encoding="utf-8") as out_file:
        line = '\t'.join(row)
        out_file.write(f"{line}\n")


if __name__ == '__main__':

    print(f'{get_curr_time()} : Download PoS tagger...')
    # download perceptron PoS tagger
    nltk.download('averaged_perceptron_tagger')

    # read British National Corpus

    bnc_texts_dir = os.path.join('storage', 'bnc', 'raw_data', 'BNC', 'Texts')
    print(f'{get_curr_time()} : Initialize BNC reader...')
    bnc_reader = BNCCorpusReader(root=bnc_texts_dir,
                                 fileids=r'[A-K]/\w*/\w*\.xml')

    # print(
    #     f'{get_curr_time()} : BNC contains {len(bnc_reader.sents())} sentences...'
    # )

    # get sentences containing MWEs from the list

    # load MWE lists
    mwe_lists_dir = os.path.join('storage', 'bnc', 'preprocessed_data')

    pmi_incorr_list_filename = 'bnc_pmi_greater_than_1_incorrect_mwe.tsv'
    dice_incorr_list_filename = 'bnc_dice_end_of_3rd_quartile_incorrect_mwe.tsv'
    chi2_incorr_list_filename = 'bnc_chi2_end_of_3rd_quartile_incorrect_mwe.tsv'

    corr_lists_filename = 'en_correct_mwe.tsv'

    for mwe_list_filename in [
            pmi_incorr_list_filename, dice_incorr_list_filename,
            chi2_incorr_list_filename, corr_lists_filename
    ][3:]:

        mwe_list_path = os.path.join(mwe_lists_dir, mwe_list_filename)
        print(
            f'{get_curr_time()} : Read incorrect MWE list for {mwe_list_filename.split("_")[1]} measure...'
        )
        incorr_mwe_df = pd.read_csv(mwe_list_path, sep='\t')

        incorr_mwe_list = [(str(first_word), str(second_word))
                           for first_word, second_word in zip(
                               incorr_mwe_df['first_word'].tolist(),
                               incorr_mwe_df['second_word'].tolist())
                           if first_word != '’' and second_word != '’']

        measure_name = mwe_list_filename.split('_')[1]

        if measure_name != 'correct':
            measure_name = f'{measure_name}_incorrect'

        write_row_to_file(os.path.join(
            'storage', 'bnc', 'preprocessed_data',
            f'{measure_name}_sentence_list.tsv'), [
                'first_word_tag', 'second_word_tag', 'first_word',
                'second_word', 'first_word_id', 'second_word_id', 'sentence'
            ],
                          append=False)
        sents_count = 0
        for sentence in bnc_reader.sents():

            for mwe_ind, incorr_mwe in enumerate(incorr_mwe_list):
                if incorr_mwe[0] in sentence:
                    first_word_id = [
                        i for i in range(len(sentence) - 1)
                        if sentence[i] == incorr_mwe[0]
                        and sentence[i + 1] == incorr_mwe[1]
                    ]

                    if first_word_id != []:  # if not first_word_id:
                        sents_count += 1
                        mwe_row = incorr_mwe_df.iloc[[mwe_ind
                                                      ]].values.tolist()[0]

                        # ommit MWEs from previous run
                        # if sents_count < 81177:
                        #     continue

                        write_row_to_file(os.path.join(
                            'storage', 'bnc', 'preprocessed_data',
                            f'{measure_name}_sentence_list.tsv'), [
                                mwe_row[0], mwe_row[1],
                                str(mwe_row[2]),
                                str(mwe_row[3]),
                                str(first_word_id[0]),
                                str(first_word_id[0] + 1), ' '.join(sentence)
                            ],
                                          append=True)

                        if sents_count % 10000 == 0 and sents_count > 0:
                            print(
                                f'{get_curr_time()} : Found {sents_count} sentences'
                            )

                        if sents_count > 200000:
                            break

            if sents_count > 200000:
                break

        # save dataframe to tsv
        # measure_name = incorr_list_filename.split('_')[1]
        # print(
        #     f"{get_curr_time()} : Save MWE list to {os.path.join('..', 'storage', 'bnc', 'preprocessed_data', f'{measure_name}_incorrect_sentence_list.tsv')}..."
        # )
        # sent_df_out_filepath = os.path.join(
        #     '..', 'storage', 'bnc', 'preprocessed_data',
        #     f'{measure_name}_incorrect_sentence_list.tsv')

        # save_sentence_df(sent_df, sent_df_out_filepath)
