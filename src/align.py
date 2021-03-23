
import argparse
import codecs
from pathlib import Path
from tqdm import tqdm

import torch

# import nltk
# from nltk.tokenize import word_tokenize
from simalign import SentenceAligner

# nltk.download('punkt')


def words_not_aligned(src_sentence, alignments):
    src_sentence = src_sentence.split()
    src_alignments = list(zip(*alignments))[0]
    indices_not_aligned = [i for i in range(
        len(src_sentence)) if i not in src_alignments]
    return [src_sentence[i] for i in indices_not_aligned]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align source and target file separated by \'|||\' using SimAlign.",
                                     epilog="example: python3 align.py path/to/corpus")
    parser.add_argument("corpus_path", type=Path)
    parser.add_argument("--src_lang", default='en', type=str)
    args = parser.parse_args()

    args.device = torch.device('cuda')

    corpus = codecs.open(args.corpus_path, 'r', 'utf-8').readlines()

    # Instancing model
    myaligner = SentenceAligner(
        model="bert", token_type="bpe", matching_methods="m", device=args.device)

    if args.src_lang == 'en':
        save_path = args.corpus_path.with_suffix('').as_posix() + "_alignment.txt"
    else:
        save_path = args.corpus_path.with_suffix('').as_posix() + "_alignment_src-pt.txt"
    print(f"File will be saved at \'{save_path}\'")
    all_align = 0
    miss_align = 0
    num_miss_align = 0
    total = 0
    with open(save_path, "w") as f:
        for l in tqdm(corpus):
            if l == '\n':
                f.write('\n')
                continue
            if args.src_lang == 'en':
                src_sentence, trg_sentence = l.split(" ||| ")
            else:
                trg_sentence, src_sentence = l.split(" ||| ")

            alignments = myaligner.get_word_aligns(
                src_sentence.strip(), trg_sentence.strip())
            alignments = list(alignments.values())[0]
            not_aligned = words_not_aligned(src_sentence, alignments)
            if not_aligned:
               miss_align += 1
               num_miss_align += len(not_aligned)
            else:
               all_align += 1
            total += 1
            f.write(' '.join('{}-{}'.format(i, j)
                             for (i, j) in alignments) + '\n')
        print(f'All align: {all_align}/{total} - {round((all_align / total) * 100, 2)}\%')
        print(f'Miss align: {miss_align}/{total} - {round((miss_align / total) * 100, 2)}\%; Averaging {round(num_miss_align / miss_align)} miss aligned tokens per miss aligned instance')