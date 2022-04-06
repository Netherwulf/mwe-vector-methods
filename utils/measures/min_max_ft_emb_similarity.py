import argparse

import numpy as np
import pandas as pd

from typing import Tuple

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from datasets.fasttext.generate_fasttext_embeddings import load_fasttext
from utils.logging.logger import log_message, init_pandas_tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_filepath',
                        help='path to the file containing data',
                        type=str)

    parser.add_argument('--mwe_list_path',
                        help='path to the file MWE list',
                        type=str)

    parser.add_argument('--out_filepath',
                        help='path to the output file',
                        type=str)

    parser.add_argument('--ft_model_path',
                        help='path to fasttext model file',
                        type=str)

    args = parser.parse_args()

    return args


def get_ft_embs_list(mwe_list, ft_model):
    return np.array([
        np.array([
            np.array([np.array(ft_model.get_word_vector(word), np.float32)])
            for word in mwe.split(' ')
        ]) for mwe in tqdm(mwe_list)
    ])


def get_similarity(mwe_emb, emb):
    if emb.shape[0] != 2:
        return -9999999.0

    mwe_1 = mwe_emb[0].reshape((1, -1))
    mwe_2 = mwe_emb[1].reshape((1, -1))

    emb_1 = emb[0].reshape((1, -1))
    emb_2 = emb[1].reshape((1, -1))

    mwe_1_emb_1_sim = cosine_similarity(mwe_1, emb_1)[0][0]
    mwe_1_emb_2_sim = cosine_similarity(mwe_1, emb_2)[0][0]

    mwe_2_emb_1_sim = cosine_similarity(mwe_2, emb_1)[0][0]
    mwe_2_emb_2_sim = cosine_similarity(mwe_2, emb_2)[0][0]

    return min(max(mwe_1_emb_1_sim, mwe_1_emb_2_sim),
               max(mwe_2_emb_1_sim, mwe_2_emb_2_sim))


def get_mwe_neighbour(mwe_emb, embs_list, mwe_list) -> Tuple[str, float]:
    log_message('calculating cosine similarity for single MWE...')
    # if mwe in mwe_list:
    #     return (mwe, -1.0)

    # mwe_words = mwe.split(' ')

    if mwe_emb.shape[0] < 2:
        return ('', 0.0)

    # word_embs_list = np.array(
    #     [np.array(ft_model.get_word_vector(word)) for word in mwe_words])

    # similarities_arr = np.array([0.0 for _ in range(embs_list.shape[0])])

    # for mwe = (mwe_1, mwe_2) and emb from list = (emb_1, emb_2)
    # for i, emb in enumerate(tqdm(embs_list)):

    sim_func = lambda emb: get_similarity(mwe_emb, emb)

    similarities_arr = np.array([
        sim_func(emb) for idx, emb in tqdm(np.ndenumerate(embs_list),
                                           total=embs_list.shape[0])
    ])

    # similarities_arr = np.array(
    #     [get_similarity(mwe_emb, emb) for emb in tqdm(embs_list)])

    # mwes_list_db = db.from_sequence(embs_list, partition_size=2)

    # similarities_arr = mwes_list_db.map(
    #     lambda emb: get_similarity(word_embs_list, emb))

    # with TqdmCallback(desc='compute'):
    #     similarities_arr = compute(similarities_arr)

    # similarities_arr = np.array([
    #     Thread(target=get_similarity, args=(
    #         word_embs_list,
    #         emb,
    #     )).start() for emb in tqdm(embs_list)
    # ])

    return (mwe_list[similarities_arr.argmax()], max(similarities_arr))


def get_row_cosine_similarity(row, mwe_list_embs, mwe_list, ft_model):
    return get_mwe_neighbour(row['mwe'], mwe_list_embs, mwe_list, ft_model)[1]


def get_row_mwe_neighbour(row, mwe_list_embs, mwe_list, ft_model):
    return get_mwe_neighbour(row['mwe'], mwe_list_embs, mwe_list, ft_model)[0]


def generate_min_max_ft_embs(data_path, mwe_list_path, out_filepath, ft_model):
    log_message('loading data...')

    df = pd.read_csv(data_path,
                     sep='\t',
                     header=None,
                     names=['combined_value', 'mwe_type', 'mwe'])

    # ddf = dd.from_pandas(df, npartitions=2)

    log_message('loading MWE list...')

    mwe_list_df = pd.read_csv(mwe_list_path,
                              sep='\t').drop_duplicates(subset=['lemma'])

    mwe_list = mwe_list_df['lemma'].tolist()

    log_message('generating fasttext embeddings')

    mwe_list_embs = get_ft_embs_list(mwe_list, ft_model)

    mwe_candidates_list = get_ft_embs_list(df['mwe'].tolist(), ft_model)

    log_message('calculating cosine similarity...')

    # mwe_neighbours_list = [
    #     get_mwe_neighbour(mwe_candidate, mwe_list_embs, mwe_list, ft_model)
    #     for mwe_candidate in tqdm(mwe_candidates_list)
    # ]

    get_cos_similarities_func = lambda mwe_candidate_emb: get_mwe_neighbour(
        mwe_candidate_emb, mwe_list_embs, mwe_list)[1]

    cos_similarities = [
        get_cos_similarities_func(mwe_candidate)
        for idx, mwe_candidate in tqdm(np.ndenumerate(mwe_candidates_list),
                                       total=mwe_candidates_list.shape[0])
    ]

    # [
    #     get_mwe_neighbour(mwe_candidate, mwe_list_embs, mwe_list, ft_model)
    #     for mwe_candidate in tqdm(mwe_candidates_list)
    # ]

    # cos_similarities = [elem[1] for elem in tqdm(mwe_neighbours_list)]

    # mwe_neighbours = [elem[0] for elem in tqdm(mwe_neighbours_list)]

    # for i, mwe_candidate in enumerate(tqdm(mwe_candidates_list)):

    #     mwe_neighbour, cos_sim = get_mwe_neighbour(mwe_candidate,
    #                                                mwe_list_embs, mwe_list,
    #                                                ft_model)

    #     cos_similarities[i] = cos_sim
    #     mwe_neighbours[i] = mwe_neighbour

    # with TqdmCallback(desc='compute'):
    #     ddf['cosine_similarity'] = ddf.apply(get_row_cosine_similarity,
    #                                          axis=1,
    #                                          args=(mwe_list_embs, mwe_list,
    #                                                ft_model)).compute()

    # df['cosine_similarity'] = np.array([
    #     get_mwe_neighbour(mwe, mwe_list_embs, mwe_list, ft_model)[1]
    #     for mwe in tqdm(df['mwe'].values)
    # ])

    # df['cosine_similarity'] = np.array([
    #     get_mwe_neighbour(mwe_candidate_emb, mwe_list_embs, mwe_list,
    #                       ft_model)[1]
    #     for mwe_candidate_emb in tqdm(mwe_candidates_list)
    # ])

    df.loc[:, 'cosine_similarity'] = df['cosine_similarity'].round(4)

    # df['mwe_neigbour'] = df.progress_apply(lambda row: get_mwe_neighbour(
    #     row['mwe'], mwe_list_embs, mwe_list, ft_model)[0],
    #                                        axis=1)

    df['mwe_neigbour'] = np.array([
        get_mwe_neighbour(mwe_candidate_emb, mwe_list_embs, mwe_list,
                          ft_model)[0]
        for mwe_candidate_emb in tqdm(mwe_candidates_list)
    ])

    log_message('add is_correct column')

    df = pd.merge(df, mwe_list_df[['lemma', 'is_correct']], on='lemma')

    log_message('saving df to tsv...')

    df = df.sort_values(by=['cosine_similarity'], ascending=[False])

    df.to_csv(out_filepath, sep='\t', index=False)


def main():
    args = parse_args()

    data_filepath = args.data_filepath
    mwe_list_path = args.mwe_list_path
    out_filepath = args.out_filepath
    ft_model_path = args.ft_model_path

    init_pandas_tqdm()

    ft_model = load_fasttext(ft_model_path)

    generate_min_max_ft_embs(data_filepath, mwe_list_path, out_filepath,
                             ft_model)


if __name__ == '__main__':
    main()
