import sys

import pandas as pd


def merge_prediction_results(filepath_list):
    for filepath_idx, filepath in enumerate(filepath_list):
        if filepath_idx == 0:
            df = pd.read_csv(filepath, sep='\t')

            df = df.rename(columns={df.columns[-1]: 'majority_cnn'})

            df = df.iloc[:, [0, 2, 4, 5, 6]]

        else:
            df_temp = pd.read_csv(filepath, sep='\t')

            model_name = filepath.split('.')[0].split('_')[-3] + '_' + filepath.split('.')[0].split('_')[-1]

            df[model_name] = df_temp['model_prediction']

    final_df = df[
        (~df['mwe'].astype(str).str.contains('po')) & (df['first_word_orth'] != 'po') & (
                df['first_word_orth'].astype(str).str.len() > 2) & (
            ~df['first_word_orth'].astype(str).str.contains('http'))]

    # final_df = df[
    #     (df['first_word_orth'] != 'po') & (df['first_word_lemma'] != 'po') & (
    #                 df['first_word_lemma'].astype(str).str.len() > 2) & (
    #             df['first_word_orth'].astype(str).str.len() > 2) & (
    #         ~df['first_word_orth'].astype(str).str.contains('http')) & (
    #         ~df['first_word_lemma'].astype(str).str.contains('http'))]

    return final_df


def main(args):
    filepath_list = ['results_transformer_embeddings_svm_smote_majority_voting_cnn.tsv',
                     'results_transformer_embeddings_svm_smote_majority_voting_lr.tsv',
                     'results_transformer_embeddings_svm_smote_majority_voting_rf.tsv',
                     'results_transformer_embeddings_svm_smote_weighted_voting_cnn.tsv',
                     'results_transformer_embeddings_svm_smote_weighted_voting_lr.tsv',
                     'results_transformer_embeddings_svm_smote_weighted_voting_rf.tsv']

    df = merge_prediction_results(filepath_list)

    df.to_csv('merged_predictions.tsv', sep='\t', encoding='utf-8', index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
