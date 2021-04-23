# %%
import argparse
from pathlib import Path
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated corpus against gold standard.",
                                     epilog="example: python3 eval_en_from_pt.py path/to/corpus path/to/gold")
    parser.add_argument("gold_standard_dir", type=Path)
    parser.add_argument("generated_corpus_dir", type=Path)
    parser.add_argument("--set_split", default='all')
    parser.add_argument("--with_pad", action='store_true', default=False)
    args = parser.parse_args()

    print(f'{"With" if args.with_pad else "Without"} Padding')

    if args.set_split == 'all':
        gold_train = open(args.gold_standard_dir / 'train.dat').readlines()
        gold_dev = open(args.gold_standard_dir / 'dev.dat').readlines()
        gold_test = open(args.gold_standard_dir / 'test.dat').readlines()
        gold_standard = []
        for split in [gold_train, gold_dev, gold_test]:
            gold_standard.extend(split)

        gen_corpus = []
        gen_train = open(args.generated_corpus_dir / f'train{"_pad" if args.with_pad else ""}.dat').readlines()
        gen_dev = open(args.generated_corpus_dir / f'dev{"_pad" if args.with_pad else ""}.dat').readlines()
        gen_test = open(args.generated_corpus_dir / f'test{"_pad" if args.with_pad else ""}.dat').readlines()
        for split in [gen_train, gen_dev, gen_test]:
            gen_corpus.extend(split)

    else:
        gold_standard = open(args.gold_standard_dir / f'{args.set_split}.dat').readlines()
        gen_corpus = open(args.generated_corpus_dir / f'{args.set_split}{"_pad" if args.with_pad else ""}.dat').readlines()
    print(f'Length Gold Standard: {len(gold_standard)}')
    print(f'Length Generated Corpus: {len(gen_corpus)}')

    tag_list = ['B-Claim', 'B-MajorClaim', 'B-Premise', 'I-Claim', 'I-MajorClaim', 'I-Premise', 'O']

    def build_msg(F1, TP, FP, FN):
            msg = f'Class{" "*8} | Support | Precision | Recall | F1\n'
            msg += '-' * 40 + '\n'
            sum_M_F1 = 0
            sum_support = 0
            sum_precision = 0
            sum_recall = 0
            for tag in tag_list:
                sum_M_F1 += F1[tag]
                support = TP[tag] + FN[tag]
                sum_support += support
                precision = TP[tag] / (TP[tag] + FP[tag])
                sum_precision += precision
                recall = TP[tag] / support
                sum_recall += recall
                msg += f'{tag:13s} | {support:7d} | {precision:9.2f} | {recall:6.2f} | {F1[tag]:1.2f}\n'
            M_F1 = sum_M_F1 / len(F1)
            sum_precision = sum_precision / len(tag_list)
            sum_recall = sum_recall / len(tag_list)
            msg += '-'*40 + '\n'
            msg += f'Overall{" "*6} | {sum_support:7d} | {sum_precision:9.2f} | {sum_recall:6.2f} | {M_F1:1.2f}\n'
            msg += '-'*40 + '\n'
            return msg

    def add_to_dict(dict_in, tag, val):
        if tag in dict_in:
            dict_in[tag] += val
        else:
            dict_in[tag] = val
        return dict_in

    # Init values
    TP = {tag: 0 for tag in tag_list}
    FP = {tag: 0 for tag in tag_list}
    FN = {tag: 0 for tag in tag_list}
    F1 = {tag: 0 for tag in tag_list}
    for gold_seq, gen_seq in zip(gold_standard, gen_corpus):
        if gold_seq == '\n':
            continue
        gold_tag_sequence = gold_seq.strip().split('\t')[-1]
        gen_tag_sequence = gen_seq.strip().split('\t')[-1]

        gold_tag_sequence = gold_tag_sequence.split(":")[0] # remove relation

        if gold_tag_sequence == gen_tag_sequence:
            TP = add_to_dict(TP, gold_tag_sequence, 1)
        else:
            FN = add_to_dict(FN, gold_tag_sequence, 1)
            FP = add_to_dict(FP, gen_tag_sequence, 1)
    # Calculate F1 for each tag
    for tag in tag_list:
        F1[tag] = (2 * TP[tag] / max(2 * TP[tag] + FP[tag] + FN[tag], 1)) * 100
    # Calculate Macro-F1 score and prepare the message
    print(build_msg(F1, TP, FP, FN))

# %%
"""
# sanity check
gold_standard = open('../data/en_pe/train.dat').readlines()
gen_corpus = open('../data/en_from_pt_pe/train.dat').readlines()
print(f'Length Gold S        gold_train = open(args.gold_standard_dir / 'train.dat').readlines()
tandard: {len(gold_standard)}')
print(f'Length Generated Corpus: {len(gen_corpus)}')

correct = 0
incorrect = 0
incorrect_by_skip = {}
total = 0
skipped_sents = 0
skip_sent = False
gold_idx = 0
for idx, row in enumerate(gold_standard):
    total += 1
    gold = row.strip().split('\t')
    gen = gen_corpus[gold_idx].strip().split('\t')

    if skip_sent:
        incorrect_by_skip[skipped_sents] = incorrect_by_skip.get(skipped_sents, 0) + 1
        if len(row_list) == 1:
            skip_sent = False
        continue

    if len(gold) == 3:
        gold[2] = gold[2].split(":")[0] # take out relation
    
    if gen == gold:
        # completely equal
        correct += 1
    else:
        if gold[:-1] == gen[:-1]:
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
"""
# %%
