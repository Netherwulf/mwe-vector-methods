import os
import re
import sys

import morfeusz2
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel


# init Morfeusz2 lemmatizer
def init_lemmatizer():
    return morfeusz2.Morfeusz()  # initialize Morfeusz object


def get_word_offset_ids(sentence, word_id, offset_mapping):
    sent_offsets = [(ele.start(), ele.end())
                    for ele in re.finditer(r'\S+', sentence)]

    try:
        word_offset = sent_offsets[word_id]
    except IndexError:
        print('IndexError: word_offset = sent_offsets[word_id]',
              f'sentence: {sentence}',
              f'word_id: {word_id}',
              f'sent_offsets: {sent_offsets}',
              f'offset_mapping: {offset_mapping[0]}',
              '\n',
              sep='\n')

        return [0]

    word_offset_mappings_ind = [
        ind for ind, elem in enumerate(offset_mapping[0])
        if elem[0] == word_offset[0] or elem[1] == word_offset[1]
    ]

    return word_offset_mappings_ind


def get_hidden_states(encoded, offset_ids, model, layers):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
       Select only those subword token outputs that belong to our word of interest
       and average them."""

    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states

    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    # Only select the tokens that constitute the requested word
    word_tokens_output = output[offset_ids]

    return word_tokens_output.mean(dim=0).to('cpu').numpy()


def get_word_vector(sent, word_id, tokenizer, model, layers):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
       that make up the word of interest, and then `get_hidden_states`."""
    encoded = tokenizer.encode_plus(sent,
                                    padding='longest',
                                    add_special_tokens=True,
                                    return_tensors="pt",
                                    return_offsets_mapping=True)
    offset_mapping = encoded['offset_mapping']

    offset_ids = get_word_offset_ids(sent, word_id, offset_mapping)

    encoded = {
        key: encoded[key]
        for key in encoded.keys() if key != 'offset_mapping'
    }

    return get_hidden_states(encoded, offset_ids, model, layers)


def create_empty_file(filepath):
    with open(filepath, 'x', buffering=2000000) as _:
        pass


def write_line_to_file(filepath, line):
    with open(filepath, 'a', buffering=2000000) as f:
        f.write(f'{line}\n')


def get_word_embedding(sentence, word_id, tokenizer, model, layers):
    word_embedding = get_word_vector(sentence, word_id, tokenizer, model,
                                     layers)

    return word_embedding  # , word_occured


def substitute_and_embed(sentence, old_word_id, new_word, tokenizer, model,
                         layers):
    # sentence_to_substitute = sentence[:]
    sentence_words = sentence.split(' ')
    # only one MWE component appears in the sentence
    if len(old_word_id) == 1:
        if old_word_id[0] == len(sentence_words) - 1:
            sentence_to_substitute = ' '.join(
                [' '.join(sentence_words[:old_word_id[0]]), new_word])

        else:
            sentence_to_substitute = ' '.join([
                ' '.join(sentence_words[:old_word_id[0]]), new_word,
                ' '.join(sentence_words[old_word_id[0] + 1:])
            ])

    # two MWE components appear in the sentence
    if len(old_word_id) == 2:
        if old_word_id[1] == len(sentence.split(' ')) - 1:
            sentence_to_substitute = ' '.join(
                [' '.join(sentence_words[:old_word_id[0]]), new_word])

        else:
            sentence_to_substitute = ' '.join([
                ' '.join(sentence_words[:old_word_id[0]]), new_word,
                ' '.join(sentence_words[old_word_id[1] + 1:])
            ])

    if len(new_word.split(' ')) > 1:
        first_word, second_word = new_word.split(' ')

        first_word_emb = get_word_embedding(sentence_to_substitute,
                                            old_word_id[0], tokenizer, model,
                                            layers)

        if len(old_word_id) > 1:
            second_word_emb = get_word_embedding(sentence_to_substitute,
                                                 old_word_id[1], tokenizer,
                                                 model, layers)
        else:
            second_word_emb = get_word_embedding(sentence_to_substitute,
                                                 old_word_id[0] + 1, tokenizer,
                                                 model, layers)

        emb = [(first_word_elem + second_word_elem) / 2
               for first_word_elem, second_word_elem in zip(
                   first_word_emb, second_word_emb)]

    else:
        emb = get_word_embedding(sentence_to_substitute, old_word_id[0],
                                 tokenizer, model, layers)

    return emb


def read_tsv(filepath, tokenizer, model, layers):
    filepath_dir = os.path.join(os.path.join(*filepath.split('/')[:-2]),
                                'embeddings', 'transformer')
    filepath_name = filepath.split('/')[-1].split('.')[0]

    complete_mwe_in_sent_output_file = os.path.join(
        filepath_dir, filepath_name +
        f'_embeddings_{len(layers)}_layers_complete_mwe_in_sent.tsv')
    create_empty_file(complete_mwe_in_sent_output_file)

    write_line_to_file(
        complete_mwe_in_sent_output_file, '\t'.join([
            'mwe_type', 'first_word', 'first_word_id', 'second_word',
            'second_word_id', 'mwe', 'sentence', 'is_correct', 'dataset_type',
            'complete_mwe_in_sent', 'mwe_embedding',
            'first_word_only_embedding', 'second_word_only_embedding',
            'first_word_mwe_emb_diff', 'second_word_mwe_emb_diff',
            'first_word_mwe_emb_abs_diff', 'second_word_mwe_emb_abs_diff',
            'first_word_mwe_emb_prod', 'second_word_mwe_emb_prod',
            'combined_embedding'
        ]))

    incomplete_mwe_in_sent_output_file = os.path.join(
        filepath_dir, filepath_name +
        f'_embeddings_{len(layers)}_layers_incomplete_mwe_in_sent.tsv')
    create_empty_file(incomplete_mwe_in_sent_output_file)

    write_line_to_file(
        incomplete_mwe_in_sent_output_file, '\t'.join([
            'mwe_type', 'first_word', 'first_word_id', 'second_word',
            'second_word_id', 'mwe', 'sentence', 'is_correct', 'dataset_type',
            'complete_mwe_in_sent', 'first_word_embedding', 'mwe_embedding',
            'first_word_mwe_emb_diff', 'first_word_mwe_emb_abs_diff',
            'first_word_mwe_emb_prod', 'combined_embedding'
        ]))

    with open(filepath, 'r', errors='replace', buffering=2000000) as in_file:
        content = in_file.readlines()

        for line in content[1:]:
            line = line.strip()

            line_attributes = line.split('\t')

            # KGR10 (Słowosieć) sentence list column mapping
            # mwe_type = 'null'
            # mwe_lemma = line_attributes[1]
            # first_word = line_attributes[5]
            # first_word_id = int(line_attributes[4])
            # second_word = line_attributes[8]
            # second_word_id = int(line_attributes[7])
            # mwe = line_attributes[0]
            # sentence = line_attributes[12]
            # is_correct = str(line_attributes[2])
            # dataset_type = 'null'
            # complete_mwe_in_sent = line_attributes[3]

            # MeWeX detected MWEs in KGR10 corpus sentence list column mapping
            # mwe_type = 'null'
            # mwe_lemma = line_attributes[1]
            # first_word = line_attributes[4]
            # first_word_id = int(line_attributes[3])
            # second_word = line_attributes[7]
            # second_word_id = int(line_attributes[6])
            # mwe = line_attributes[0]
            # sentence = line_attributes[11]
            # is_correct = 'null'
            # dataset_type = 'null'
            # complete_mwe_in_sent = line_attributes[2]

            # PARSEME sentence list column mapping
            # mwe_type = line_attributes[0]
            # first_word = line_attributes[1]
            # first_word_id = int(line_attributes[3])
            # second_word = line_attributes[4]
            # second_word_id = int(line_attributes[6])
            # mwe = line_attributes[7]
            # sentence = line_attributes[9]
            # is_correct = str(line_attributes[10])
            # dataset_type = line_attributes[11]
            # complete_mwe_in_sent = '1'

            # BNC sentence list column mapping
            mwe_type = f'{line_attributes[0]}+{line_attributes[1]}'
            first_word = line_attributes[2]
            first_word_id = int(line_attributes[4])
            second_word = line_attributes[3]
            second_word_id = int(line_attributes[5])
            mwe = f'{line_attributes[2]} {line_attributes[3]}'
            sentence = line_attributes[6]
            is_correct = '0'
            dataset_type = 'null'
            complete_mwe_in_sent = '1'

            # complete MWE appears in the sentence
            if complete_mwe_in_sent == '1':
                # try:
                first_word_embedding = get_word_embedding(
                    sentence, first_word_id, tokenizer, model, layers)
                second_word_embedding = get_word_embedding(
                    sentence, second_word_id, tokenizer, model, layers)
                # except RuntimeError:
                #     print(
                #         f'embedding too short for words: {first_word} - {second_word}'
                #     )
                #     continue

                mwe_embedding = [
                    (first_word_elem + second_word_elem) / 2
                    for first_word_elem, second_word_elem in zip(
                        first_word_embedding, second_word_embedding)
                ]

                # try:
                first_word_only_embedding = substitute_and_embed(
                    sentence, [first_word_id, second_word_id], first_word,
                    tokenizer, model, layers)

                second_word_only_embedding = substitute_and_embed(
                    sentence, [first_word_id, second_word_id], second_word,
                    tokenizer, model, layers)

                # except RuntimeError:
                #     print(
                #         f'embedding too short for sentence and words: {sentence} - {first_word} - {second_word}'
                #     )
                #     continue

                first_word_mwe_emb_diff = [
                    str(mwe_elem - first_word_elem)
                    for mwe_elem, first_word_elem in zip(
                        mwe_embedding, first_word_only_embedding)
                ]

                second_word_mwe_emb_diff = [
                    str(mwe_elem - second_word_elem)
                    for mwe_elem, second_word_elem in zip(
                        mwe_embedding, second_word_only_embedding)
                ]

                first_word_mwe_emb_abs_diff = [
                    str(abs(mwe_elem - first_word_elem))
                    for mwe_elem, first_word_elem in zip(
                        mwe_embedding, first_word_only_embedding)
                ]

                second_word_mwe_emb_abs_diff = [
                    str(abs(mwe_elem - second_word_elem))
                    for mwe_elem, second_word_elem in zip(
                        mwe_embedding, second_word_only_embedding)
                ]

                first_word_mwe_emb_prod = [
                    str(mwe_elem * first_word_elem)
                    for mwe_elem, first_word_elem in zip(
                        mwe_embedding, first_word_only_embedding)
                ]

                second_word_mwe_emb_prod = [
                    str(mwe_elem * second_word_elem)
                    for mwe_elem, second_word_elem in zip(
                        mwe_embedding, second_word_only_embedding)
                ]

                mwe_embedding = [str(elem) for elem in mwe_embedding]

                first_word_only_embedding = [
                    str(elem) for elem in first_word_only_embedding
                ]

                second_word_only_embedding = [
                    str(elem) for elem in second_word_only_embedding
                ]

                combined_embedding = np.hstack(
                    (mwe_embedding, first_word_only_embedding,
                     second_word_only_embedding, first_word_mwe_emb_diff,
                     second_word_mwe_emb_diff, first_word_mwe_emb_abs_diff,
                     second_word_mwe_emb_abs_diff, first_word_mwe_emb_prod,
                     second_word_mwe_emb_prod))

                write_line_to_file(
                    complete_mwe_in_sent_output_file, '\t'.join([
                        mwe_type, first_word,
                        str(first_word_id), second_word,
                        str(second_word_id), mwe, sentence, is_correct,
                        dataset_type, complete_mwe_in_sent,
                        ','.join(mwe_embedding),
                        ','.join(first_word_only_embedding),
                        ','.join(second_word_only_embedding),
                        ','.join(first_word_mwe_emb_diff),
                        ','.join(second_word_mwe_emb_diff),
                        ','.join(first_word_mwe_emb_abs_diff),
                        ','.join(second_word_mwe_emb_abs_diff),
                        ','.join(first_word_mwe_emb_prod),
                        ','.join(second_word_mwe_emb_prod),
                        ','.join(combined_embedding)
                    ]))

            # only part of MWE appears in the sentence
            else:
                # try:
                first_word_embedding = get_word_embedding(
                    sentence, first_word_id, tokenizer, model, layers)

                mwe_embedding = substitute_and_embed(sentence, [first_word_id],
                                                     mwe, tokenizer, model,
                                                     layers)

                # except RuntimeError:
                #     print(
                #         f'embedding too short for sentence and id: {sentence} - {first_word_id}'
                #     )
                #     continue

                first_word_mwe_emb_diff = [
                    str(mwe_elem - first_word_elem)
                    for mwe_elem, first_word_elem in zip(
                        mwe_embedding, first_word_embedding)
                ]

                first_word_mwe_emb_abs_diff = [
                    str(abs(mwe_elem - first_word_elem))
                    for mwe_elem, first_word_elem in zip(
                        mwe_embedding, first_word_embedding)
                ]

                first_word_mwe_emb_prod = [
                    str(mwe_elem * first_word_elem)
                    for mwe_elem, first_word_elem in zip(
                        mwe_embedding, first_word_embedding)
                ]

                first_word_embedding = [
                    str(elem) for elem in first_word_embedding
                ]

                mwe_embedding = [str(elem) for elem in mwe_embedding]

                combined_embedding = np.hstack(
                    (first_word_embedding, mwe_embedding,
                     first_word_mwe_emb_diff, first_word_mwe_emb_abs_diff,
                     first_word_mwe_emb_prod))

                write_line_to_file(
                    incomplete_mwe_in_sent_output_file, '\t'.join([
                        mwe_type, first_word,
                        str(first_word_id), second_word,
                        str(second_word_id), mwe, sentence, is_correct,
                        dataset_type, complete_mwe_in_sent,
                        ','.join(first_word_embedding),
                        ','.join(mwe_embedding),
                        ','.join(first_word_mwe_emb_diff),
                        ','.join(first_word_mwe_emb_abs_diff),
                        ','.join(first_word_mwe_emb_prod),
                        ','.join(combined_embedding)
                    ]))


def main(args):
    model_name = 'xlm-roberta-base'  # allegro/herbert-base-cased
    layers = 1  # layers = 4

    layers = [layer_num for layer_num in range(-1 * layers - 1, -1, 1)
              ]  # [-2] or [-3, -2] or [-4, -3, -2] or more

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    use_cuda = False

    device = "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"
    model = model.to(device)

    # lemmatizer = init_lemmatizer()

    for filepath in args:
        read_tsv(filepath, tokenizer, model, layers)


if __name__ == '__main__':
    main(sys.argv[1:])
