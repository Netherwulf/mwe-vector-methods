import sys

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel


def get_word_idx(sent: str, word: str):
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

    return get_word_embedding(sentence, new_word, tokenizer, model, layers)


def read_tsv(filepath, tokenizer, model, layers):
    complete_mwe_in_sent_output_file = filepath.split('/')[-1].split('.')[
                                           0] + f'_embeddings_{len(layers)}_layers_complete_mwe_in_sent.tsv'
    create_empty_file(complete_mwe_in_sent_output_file)

    incomplete_mwe_in_sent_output_file = filepath.split('/')[-1].split('.')[
                                             0] + f'_embeddings_{len(layers)}_layers_incomplete_mwe_in_sent.tsv'
    create_empty_file(incomplete_mwe_in_sent_output_file)

    with open(filepath, 'r', errors='replace') as in_file:
        content = in_file.readlines()

        for line in content:
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
                write_line_to_file(complete_mwe_in_sent_output_file, '')

            # only part of MWE appears in the sentence
            else:
                write_line_to_file(incomplete_mwe_in_sent_output_file, '')


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
