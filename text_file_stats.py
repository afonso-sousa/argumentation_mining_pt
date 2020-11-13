import argparse
from pathlib import Path

import nltk.data
from nltk import word_tokenize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get stats from free text file.",
                                     epilog="example: python3 text_file_stats.py path/to/corpus")
    parser.add_argument("corpus_path", type=Path)

    args = parser.parse_args()

    # Load data
    nltk_text = nltk.data.load(args.corpus_path.as_posix())

    print('# of tokens: {}'.format(len(nltk_text)))

    tokens = word_tokenize(nltk_text)
    print('Vocab length: {}'.format(len(tokens)))

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(nltk_text)
    print('# of sentences: {}'.format(len(sentences)))
