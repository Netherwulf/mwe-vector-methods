import re
import sys

import morfeusz2
import torch

from transformers import AutoTokenizer, AutoModel


# init Morfeusz2 lemmatizer
def init_lemmatizer():
    return morfeusz2.Morfeusz()  # initialize Morfeusz object


# def get_word_idx(sent: str, word: str):  # (sent: str, word: str, lemmatizer)
#     return sent.index(word)
#     sentence = sent[:]
#     sentence = [str(lemmatizer.analyse(word_elem)[0][2][1]) if word_elem not in string.punctuation else word_elem for
#                 word_elem in sentence.split(' ')]
#     word_lemma = lemmatizer.analyse(word)[0][2][1]
#
#     if word_lemma in sentence:
#         return sentence.index(word_lemma), True
#
#     else:
#         print(f'word: {word_lemma} doesnt occur in sentence: \n{sentence}')
#         return -1, False


def get_word_offset_ids(sentence, word_id, offset_mapping):
    sent_offsets = [(ele.start(), ele.end()) for ele in re.finditer(r'\S+', sentence)]
    # print(f'sentence = {sentence}',
    # f'sent_offsets = {sent_offsets}',
    # f'word_id = {word_id}',
    # sep='\n')
    word_offset = sent_offsets[word_id]
    # print(f'word_offset = {word_offset}',
    #       f'offset_mapping = {offset_mapping}',
    #       sep='\n')
    word_offset_mappings_ind = [ind for ind, elem in enumerate(offset_mapping[0]) if
                                elem[0] == word_offset[0] or elem[1] == word_offset[1]]

    # print(f'sentence = {sentence}',
    # f'word_id = {word_id}',
    # f'sent_offsets = {sent_offsets}',
    # f'word_offset = {word_offset}',
    # f'offset_mapping = {offset_mapping}',
    # f'word_offset_mappings_ind = {word_offset_mappings_ind}',
    # sep='\n')

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
    # print(f'states = {states}',
    #       f'output = {output}',
    #       f'word_tokens_output = {word_tokens_output}',
    #       sep='\n')
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

    encoded = {key: encoded[key] for key in encoded.keys() if key != 'offset_mapping'}
    # get all token idxs that belong to the word of interest
    # token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
    # print(f'encoded KEYS = {encoded.keys()}',
    # f'offset_mapping = {offset_mapping}',
    # f'word_id = {encoded["input_ids"]}',
    # f'decoded sentence = {tokenizer.decode(encoded["input_ids"][0])}',
    # sep='\n')
    return get_hidden_states(encoded, offset_ids, model, layers)


def create_empty_file(filepath):
    with open(filepath, 'w') as _:
        pass


def write_line_to_file(filepath, line):
    with open(filepath, 'a') as f:
        f.write(f'{line}\n')


def get_word_embedding(sentence, word_id, tokenizer, model, layers):
    # idx = get_word_idx(sentence, word)  # (sentence, word, lemmatizer)
    # idx, word_occured = get_word_idx(sentence, word, lemmatizer)

    word_embedding = get_word_vector(sentence, word_id, tokenizer, model, layers)
    # print(f'word id = {word_id}',
    # f'word embedding = {word_embedding}',
    # sep='\n')
    return word_embedding  # , word_occured


def substitute_and_embed(sentence, old_word_id, new_word, tokenizer, model, layers):
    # sentence_to_substitute = sentence[:]
    sentence_words = sentence.split(' ')
    # only one MWE component appears in the sentence
    if len(old_word_id) == 1:
        if old_word_id[0] == len(sentence_words) - 1:
            sentence_to_substitute = ' '.join([' '.join(sentence_words[:old_word_id[0]]), new_word])

        else:
            sentence_to_substitute = ' '.join(
                [' '.join(sentence_words[:old_word_id[0]]), new_word, ' '.join(sentence_words[old_word_id[0] + 1:])])

    # two MWE components appear in the sentence
    if len(old_word_id) == 2:
        if old_word_id[1] == len(sentence.split(' ')) - 1:
            sentence_to_substitute = ' '.join([' '.join(sentence_words[:old_word_id[0]]), new_word])

        else:
            sentence_to_substitute = ' '.join(
                [' '.join(sentence_words[:old_word_id[0]]), new_word, ' '.join(sentence_words[old_word_id[1] + 1:])])

    # print(f'sentence = {sentence}',
    # f'sentence_to_substitute = {sentence_to_substitute}',
    # sep='\n')

    if len(new_word.split(' ')) > 1:
        first_word, second_word = new_word.split(' ')

        first_word_emb = get_word_embedding(sentence_to_substitute, old_word_id[0], tokenizer, model, layers)
        # first_word_emb, word_occured = get_word_embedding(sentence, first_word, tokenizer, model, layers, lemmatizer)

        # if not word_occured:
        #     return False

        second_word_emb = get_word_embedding(sentence_to_substitute, old_word_id[1], tokenizer, model, layers)
        # second_word_emb, word_occured = get_word_embedding(sentence, second_word, tokenizer, model, layers, lemmatizer)

        # if not word_occured:
        #     return False

        emb = [(first_word_elem + second_word_elem) / 2 for first_word_elem, second_word_elem in
               zip(first_word_emb, second_word_emb)]

    else:
        emb = get_word_embedding(sentence_to_substitute, old_word_id[0], tokenizer, model, layers)

    return emb


def read_tsv(filepath, tokenizer, model, layers):
    filepath_name = filepath.split('/')[-1].split('.')[0]

    complete_mwe_in_sent_output_file = filepath_name + f'_embeddings_{len(layers)}_layers_complete_mwe_in_sent.tsv'
    create_empty_file(complete_mwe_in_sent_output_file)

    write_line_to_file(complete_mwe_in_sent_output_file, '\t'.join(
        ['mwe_type', 'first_word', 'first_word_id', 'second_word', 'second_word_id', 'mwe', 'sentence',
         'is_correct', 'complete_mwe_in_sent', 'mwe_embedding', 'first_word_only_embedding',
         'second_word_only_embedding', 'first_word_mwe_emb_diff', 'second_word_mwe_emb_diff',
         'first_word_mwe_emb_abs_diff', 'second_word_mwe_emb_abs_diff', 'first_word_mwe_emb_prod',
         'second_word_mwe_emb_prod']))

    incomplete_mwe_in_sent_output_file = filepath_name + f'_embeddings_{len(layers)}_layers_incomplete_mwe_in_sent.tsv'
    create_empty_file(incomplete_mwe_in_sent_output_file)

    write_line_to_file(incomplete_mwe_in_sent_output_file, '\t'.join(
        ['mwe_type', 'first_word', 'first_word_id', 'second_word', 'second_word_id', 'mwe', 'sentence',
         'is_correct', 'complete_mwe_in_sent', 'first_word_embedding', 'mwe_embedding', 'first_word_mwe_emb_diff',
         'first_word_mwe_emb_abs_diff', 'first_word_mwe_emb_prod']))

    with open(filepath, 'r', errors='replace') as in_file:
        content = in_file.readlines()

        for line in content[1:]:
            line = line.strip()

            line_attributes = line.split('\t')

            # KGR10 (Słowosieć) sentence list column mapping
            # mwe = line_attributes[0]
            # mwe_lemma = line_attributes[1]
            # is_correct = line_attributes[2]
            # complete_mwe_in_sent = line_attributes[3]
            # first_word = line_attributes[5]
            # second_word = line_attributes[8]
            # sentence = line_attributes[12]

            # PARSEME sentence list column mapping
            mwe_type = line_attributes[0]
            first_word = line_attributes[1]
            first_word_id = int(line_attributes[3])
            second_word = line_attributes[4]
            second_word_id = int(line_attributes[6])
            mwe = line_attributes[7]
            sentence = line_attributes[9]
            is_correct = str(line_attributes[10])
            complete_mwe_in_sent = '1'

            # print(f'mwe_type = {mwe_type}',
            # f'first_word = {first_word}',
            # f'first_word_id = {first_word_id}',
            # f'second_word = {second_word}',
            # f'second_word_id = {second_word_id}',
            # f'mwe = {mwe}',
            # f'sentence = {sentence}',
            # f'is_correct = {is_correct}',
            # f'complete_mwe_in_sent = {complete_mwe_in_sent}',
            # sep='\n')

            # complete MWE appears in the sentence
            if complete_mwe_in_sent == '1':
                first_word_embedding = get_word_embedding(sentence, first_word_id, tokenizer, model, layers)
                second_word_embedding = get_word_embedding(sentence, second_word_id, tokenizer, model, layers)

                mwe_embedding = [(first_word_elem + second_word_elem) / 2 for first_word_elem, second_word_elem in
                                 zip(first_word_embedding, second_word_embedding)]

                first_word_only_embedding = substitute_and_embed(sentence, [first_word_id, second_word_id], first_word,
                                                                 tokenizer, model, layers)

                second_word_only_embedding = substitute_and_embed(sentence, [first_word_id, second_word_id],
                                                                  second_word, tokenizer, model, layers)

                first_word_mwe_emb_diff = [str(mwe_elem - first_word_elem) for mwe_elem, first_word_elem in
                                           zip(mwe_embedding, first_word_only_embedding)]

                second_word_mwe_emb_diff = [str(mwe_elem - second_word_elem) for mwe_elem, second_word_elem in
                                            zip(mwe_embedding, second_word_only_embedding)]

                first_word_mwe_emb_abs_diff = [str(abs(mwe_elem - first_word_elem)) for mwe_elem, first_word_elem in
                                               zip(mwe_embedding, first_word_only_embedding)]

                second_word_mwe_emb_abs_diff = [str(abs(mwe_elem - second_word_elem)) for mwe_elem, second_word_elem in
                                                zip(mwe_embedding, second_word_only_embedding)]

                first_word_mwe_emb_prod = [str(mwe_elem * first_word_elem) for mwe_elem, first_word_elem in
                                           zip(mwe_embedding, first_word_only_embedding)]

                second_word_mwe_emb_prod = [str(mwe_elem * second_word_elem) for mwe_elem, second_word_elem in
                                            zip(mwe_embedding, second_word_only_embedding)]

                mwe_embedding = [str(elem) for elem in mwe_embedding]
                first_word_only_embedding = [str(elem) for elem in first_word_only_embedding]
                second_word_only_embedding = [str(elem) for elem in second_word_only_embedding]

                write_line_to_file(complete_mwe_in_sent_output_file, '\t'.join(
                    [mwe_type, first_word, str(first_word_id), second_word, str(second_word_id), mwe, sentence,
                     is_correct, complete_mwe_in_sent, ','.join(mwe_embedding), ','.join(first_word_only_embedding),
                     ','.join(second_word_only_embedding), ','.join(first_word_mwe_emb_diff),
                     ','.join(second_word_mwe_emb_diff), ','.join(first_word_mwe_emb_abs_diff),
                     ','.join(second_word_mwe_emb_abs_diff), ','.join(first_word_mwe_emb_prod),
                     ','.join(second_word_mwe_emb_prod)]))

            # only part of MWE appears in the sentence
            else:
                first_word_embedding = get_word_embedding(sentence, [first_word_id], tokenizer, model, layers)

                mwe_embedding = substitute_and_embed(sentence, [first_word_id], mwe, tokenizer, model, layers)

                first_word_mwe_emb_diff = [str(mwe_elem - first_word_elem) for mwe_elem, first_word_elem in
                                           zip(mwe_embedding, first_word_embedding)]

                first_word_mwe_emb_abs_diff = [str(abs(mwe_elem - first_word_elem)) for mwe_elem, first_word_elem in
                                               zip(mwe_embedding, first_word_embedding)]

                first_word_mwe_emb_prod = [str(mwe_elem * first_word_elem) for mwe_elem, first_word_elem in
                                           zip(mwe_embedding, first_word_embedding)]

                first_word_embedding = [str(elem) for elem in first_word_embedding]
                mwe_embedding = [str(elem) for elem in mwe_embedding]

                write_line_to_file(incomplete_mwe_in_sent_output_file, '\t'.join(
                    [mwe_type, first_word, int(first_word_id), second_word, int(second_word_id), mwe, sentence,
                     is_correct, complete_mwe_in_sent, ','.join(first_word_embedding), ','.join(mwe_embedding),
                     ','.join(first_word_mwe_emb_diff), ','.join(first_word_mwe_emb_abs_diff),
                     ','.join(first_word_mwe_emb_prod)]))


def main(args):
    # Use last four layers by default
    # layers = [-4, -3, -2, -1] if layers is None else layers
    model_name = 'allegro/herbert-base-cased'  # bet-base-cased
    layers = 1  # layers = 4

    layers = [layer_num for layer_num in range(-1 * layers - 1, -1, 1)]  # [-2] or [-3, -2] or [-4, -3, -2] or more

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    # lemmatizer = init_lemmatizer()

    for filepath in args:
        read_tsv(filepath, tokenizer, model, layers)


if __name__ == '__main__':
    main(sys.argv[1:])
