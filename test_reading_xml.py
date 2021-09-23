import sys
import xml.etree.ElementTree as ET
from typing import List


# read plain text from XML file
def read_xml(filepath) -> (List, List):
    tree = ET.parse(filepath)
    sentences = tree.findall('.//sentence')

    orths = []
    lemmas = []

    for sentence in sentences:
        sentence_orths = []
        sentence_lemmas = []

        for sentence_token in sentence:
            for token_attr in sentence_token:
                if token_attr.tag == 'orth':
                    sentence_orths.append(token_attr.text)

                elif token_attr.tag == 'lex':
                    for lex_attr in token_attr:
                        if lex_attr.tag == 'base':
                            sentence_lemmas.append(lex_attr.text)

        orths.append(sentence_orths)
        lemmas.append(sentence_lemmas)

    return orths, lemmas


def main(args):
    for arg in args:
        print(read_xml(arg))


if __name__ == '__main__':
    main(sys.argv[1:])
