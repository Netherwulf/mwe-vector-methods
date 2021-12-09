import re
import string
import sys

import morfeusz2
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel


# init Morfeusz2 lemmatizer
def init_lemmatizer():
    return morfeusz2.Morfeusz()  # initialize Morfeusz object


def get_word_idx(sent: str, word: str):  # (sent: str, word: str, lemmatizer)
    return sent.index(word)
    # sentence = sent[:]
    # sentence = [str(lemmatizer.analyse(word_elem)[0][2][1]) if word_elem not in string.punctuation else word_elem for
    #             word_elem in sentence.split(' ')]
    # word_lemma = lemmatizer.analyse(word)[0][2][1]
    #
    # if word_lemma in sentence:
    #     return sentence.index(word_lemma), True
    #
    # else:
    #     print(f'word: {word_lemma} doesnt occur in sentence: \n{sentence}')
    #     return -1, False


def get_word_offset_ids(sentence, word_id, offset_mapping):
    sent_offsets = [(ele.start(), ele.end()) for ele in re.finditer(r'\S+', sentence)]

    word_offset = sent_offsets[word_id]

    word_offset_mappings_ind = [ind for ind, elem in enumerate(offset_mapping) if
                                elem[0] == word_offset[0] or elem[1] == word_offset[1]]

    print(f'sentence = {sentence}',
          f'word_id = {word_id}',
          f'sent_offsets = {sent_offsets}',
          f'word_offset = {word_offset}',
          f'word_offset_mappings_ind = {word_offset_mappings_ind}',
          sep='\n')

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
    print(f'states = {states}',
          f'output = {output}',
          f'word_tokens_output = {word_tokens_output}',
          sep='\n')
    return word_tokens_output.mean(dim=0)


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

    encoded = {key: encoded[key] for key in encoded.keys() if key != 'offset_mapping'}
    # get all token idxs that belong to the word of interest
    # token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
    print(f'encoded KEYS = {encoded.keys()}',
          f'offset_mapping = {offset_mapping}',
          f'input_ids = {encoded["input_ids"]}',
          f'decoded sentence = {tokenizer.decode(encoded["input_ids"][0])}',
          sep='\n')
    return get_hidden_states(encoded, offset_ids, model, layers)


def create_empty_file(filepath):
    with open(filepath, 'w') as _:
        pass


def write_line_to_file(filepath, line):
    with open(filepath, 'a') as f:
        f.write(f'{line}\n')


def get_word_embedding(sentence, word_id, tokenizer, model, layers, lemmatizer):
    # idx = get_word_idx(sentence, word)  # (sentence, word, lemmatizer)
    # idx, word_occured = get_word_idx(sentence, word, lemmatizer)

    word_embedding = get_word_vector(sentence, word_id, tokenizer, model, layers)
    print(f'word id = {word_id}',
          f'word embedding = {word_embedding}',
          sep='\n')
    return word_embedding  # , word_occured


def substitute_and_embed(sentence, old_word, new_word, tokenizer, model, layers, lemmatizer):
    sentence_to_substitute = sentence
    sentence_to_substitute = sentence_to_substitute.replace(old_word, new_word)

    if len(new_word.split(' ')) > 1:
        first_word, second_word = new_word.split(' ')

        first_word_emb = get_word_embedding(sentence_to_substitute, first_word, tokenizer, model, layers, lemmatizer)
        # first_word_emb, word_occured = get_word_embedding(sentence, first_word, tokenizer, model, layers, lemmatizer)

        # if not word_occured:
        #     return False

        second_word_emb = get_word_embedding(sentence_to_substitute, second_word, tokenizer, model, layers,
                                             lemmatizer)
        # second_word_emb, word_occured = get_word_embedding(sentence, second_word, tokenizer, model, layers, lemmatizer)

        # if not word_occured:
        #     return False

        emb = [(first_word_elem + second_word_elem) / 2 for first_word_elem, second_word_elem in
               zip(first_word_emb, second_word_emb)]

    else:
        emb = get_word_embedding(sentence_to_substitute, new_word, tokenizer, model, layers, lemmatizer)

    return emb


def main(args):
    # Use last four layers by default
    # layers = [-4, -3, -2, -1] if layers is None else layers
    model_name = 'allegro/herbert-base-cased'  # bet-base-cased
    layers = 1  # layers = 4

    layers = [layer_num for layer_num in range(-1 * layers - 1, -1, 1)]  # [-2] or [-3, -2] or [-4, -3, -2] or more

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    lemmatizer = init_lemmatizer()

    sentence = 'Ala ma, kota ala.'
    old_word = 'kota'
    new_word = 'banan'

    sentence = sentence.replace(old_word, new_word)
    print(f'sentence = {sentence}',
          f'new_word = {new_word}',
          sep='\n')
    get_word_embedding(sentence, new_word, tokenizer, model, layers, lemmatizer)


if __name__ == '__main__':
    main(sys.argv[1:])
