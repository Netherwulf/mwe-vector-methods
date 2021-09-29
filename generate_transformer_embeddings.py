import sys

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel


def get_word_idx(sent: str, word: str):
    print(f'{sent}\n{word}')
    return sent.split(" ").index(word)


def get_hidden_states(encoded, token_ids_word, model, layers):
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
    word_tokens_output = output[token_ids_word]

    return word_tokens_output.mean(dim=0)


def get_word_vector(sent, idx, tokenizer, model, layers):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
       that make up the word of interest, and then `get_hidden_states`."""
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")
    # get all token idxs that belong to the word of interest
    token_ids_word = np.where(np.array(encoded.word_ids()) == idx)

    return get_hidden_states(encoded, token_ids_word, model, layers)


def create_empty_file(filepath):
    with open(filepath, 'w') as _:
        pass


def write_line_to_file(filepath, line):
    with open(filepath, 'a') as f:
        f.write(f'{line}\n')


def get_word_embedding(sentence, word, tokenizer, model, layers):
    idx = get_word_idx(sentence, word)

    word_embedding = get_word_vector(sentence, idx, tokenizer, model, layers)

    return word_embedding


def substitute_and_embed(sentence, old_word, new_word, tokenizer, model, layers):
    sentence = sentence.replace(old_word, new_word)

    if len(new_word.split(' ')) > 1:
        first_word, second_word = new_word.split(' ')

        first_word_emb = get_word_embedding(sentence, first_word, tokenizer, model, layers)
        second_word_emb = get_word_embedding(sentence, second_word, tokenizer, model, layers)

        emb = [(first_word_elem + second_word_elem) / 2 for first_word_elem, second_word_elem in
               zip(first_word_emb, second_word_emb)]

    else:
        emb = get_word_embedding(sentence, new_word, tokenizer, model, layers)

    return emb


def read_tsv(filepath, tokenizer, model, layers):
    filepath_name = filepath.split('/')[-1].split('.')[0]

    complete_mwe_in_sent_output_file = filepath_name + f'_embeddings_{len(layers)}_layers_complete_mwe_in_sent.tsv'
    create_empty_file(complete_mwe_in_sent_output_file)

    incomplete_mwe_in_sent_output_file = filepath_name + f'_embeddings_{len(layers)}_layers_incomplete_mwe_in_sent.tsv'
    create_empty_file(incomplete_mwe_in_sent_output_file)

    with open(filepath, 'r', errors='replace') as in_file:
        content = in_file.readlines()

        for line in content[1:]:
            line = line.strip()

            line_attributes = line.split('\t')

            mwe = line_attributes[0]
            is_correct = line_attributes[2]
            complete_mwe_in_sent = line_attributes[3]
            first_word = line_attributes[5]
            second_word = line_attributes[7]
            sentence = line_attributes[10]

            # complete MWE appears in the sentence
            if complete_mwe_in_sent == '1':
                first_word_embedding = get_word_embedding(sentence, first_word, tokenizer, model, layers)
                second_word_embedding = get_word_embedding(sentence, second_word, tokenizer, model, layers)

                mwe_embedding = [(first_word_elem + second_word_elem) / 2 for first_word_elem, second_word_elem in
                                 zip(first_word_embedding, second_word_embedding)]

                first_word_only_embedding = substitute_and_embed(sentence, mwe, first_word, tokenizer, model, layers)
                second_word_only_embedding = substitute_and_embed(sentence, mwe, second_word, tokenizer, model, layers)

                first_word_mwe_emb_diff = [mwe_elem - first_word_elem for mwe_elem, first_word_elem in
                                           zip(mwe_embedding, first_word_only_embedding)]

                second_word_mwe_emb_diff = [mwe_elem - second_word_elem for mwe_elem, second_word_elem in
                                            zip(mwe_embedding, second_word_only_embedding)]

                write_line_to_file(complete_mwe_in_sent_output_file, '\t'.join(
                    [mwe_embedding, first_word_only_embedding, second_word_only_embedding, first_word_mwe_emb_diff,
                     second_word_mwe_emb_diff, is_correct]))

            # only part of MWE appears in the sentence
            else:
                first_word_embedding = get_word_embedding(sentence, first_word, tokenizer, model, layers)

                mwe_embedding = substitute_and_embed(sentence, first_word, mwe, tokenizer, model, layers)

                first_word_mwe_emb_diff = [mwe_elem - first_word_elem for mwe_elem, first_word_elem in
                                           zip(mwe_embedding, first_word_embedding)]

                write_line_to_file(complete_mwe_in_sent_output_file, '\t'.join(
                    [first_word_embedding, mwe_embedding, first_word_mwe_emb_diff, is_correct]))


def main(args):
    # Use last four layers by default
    # layers = [-4, -3, -2, -1] if layers is None else layers
    model_name = 'allegro/herbert-base-cased'  # bet-base-cased
    layers = 1  # layers = 4

    layers = [layer_num for layer_num in range(-1 * layers - 1, -1, 1)]  # [-2] or [-3, -2] or [-4, -3, -2] or more

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    for filepath in args:
        read_tsv(filepath, tokenizer, model, layers)


if __name__ == '__main__':
    main(sys.argv[1:])
