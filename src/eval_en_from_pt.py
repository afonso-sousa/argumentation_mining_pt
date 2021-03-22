import argparse
from pathlib import Path
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated corpus against gold standard.",
                                     epilog="example: python3 eval_en_from_pt.py path/to/corpus path/to/gold")
    parser.add_argument("generated_corpus_dir", type=Path)
    parser.add_argument("gold_standard_dir", type=Path)
    args = parser.parse_args()

    gen_corpus = open(args.generated_corpus_dir / 'dev.dat').readlines()
    gold_standard = open(args.gold_standard_dir / 'dev.dat').readlines()
    print(f'Length Generated Corpus: {len(gen_corpus)}')
    print(f'Length Gold Standard: {len(gold_standard)}')

    correct = 0
    incorrect = 0
    incorrect_by_skip = {}
    total = 0
    skipped_sents = 0
    skip_sent = False
    gold_idx = 0
    for idx, row in enumerate(gen_corpus):
        total += 1
        row_list = row.strip().split('\t')
        gold = gold_standard[gold_idx].strip().split('\t')

        if skip_sent:
            incorrect_by_skip[skipped_sents] = incorrect_by_skip.get(skipped_sents, 0) + 1
            if len(row_list) == 1:
                skip_sent = False
            continue

        if len(gold) == 3:
            gold[2] = gold[2].split(":")[0] # take out relation
        
        if row_list == gold:
            # completely equal
            correct += 1
        else:
            if gold[:-1] == row_list[:-1]:
                # same token, different labels
                incorrect += 1
            else:
                print(idx)
                sys.exit(1)
                # sentence skipped by corpus creation script
                skip_sent = True
                skipped_sents += 1
                incorrect_by_skip[skipped_sents] = incorrect_by_skip.get(skipped_sents, 0) + 1
        gold_idx += 1
    print(f'Correct: {correct}')
    print(f'Incorrect: {incorrect}')
    print(f'Total: {total}')
    print(f'Skip sentences: {skipped_sents}')
    print(f'Incorrect by skip: {sum(incorrect_by_skip.values())}')
    # print(incorrect_by_skip)