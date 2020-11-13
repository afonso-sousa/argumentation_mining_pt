
import argparse
import codecs
from pathlib import Path

from simalign.simalign import SentenceAligner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align source and target file separated by \'|||\' using SimAlign.",
                                     epilog="example: python3 align.py path/to/corpus")
    parser.add_argument("corpus_path", type=Path)
    args = parser.parse_args()

    args.device = 'cuda'

    corpus = codecs.open(args.corpus_path, 'r', 'utf-8').readlines()

    # Instancing model
    myaligner = SentenceAligner(
        model="bert", token_type="bpe", matching_methods="a", device=args.device)

    save_path = args.corpus_path.stem + "_alignment.txt"
    with open(save_path, "w") as f:
        for l in corpus:
            src_sentence, trg_sentence = l.split(" ||| ")
            alignments = myaligner.get_word_aligns(src_sentence, trg_sentence)
            alignments = list(alignments.values())[0]
            # print(' '.join('{}-{}'.format(i, j) for (i, j) in alignments))
            f.write(' '.join('{}-{}'.format(i, j)
                             for (i, j) in alignments) + '\n')
