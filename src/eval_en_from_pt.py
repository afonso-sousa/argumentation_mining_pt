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

    """
    # sanity check
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

    tag_list = ['B-Premise', 'I-Premise', 'B-MajorClaim', 'I-MajorClaim', 'B-Claim', 'I-Claim', 'O']

    def get_M_F1_msg(F1):
            msg = '\nF1 scores\n'
            msg += '-' * 24 + '\n'
            sum_M_F1 = 0
            for tag in tag_list:
                sum_M_F1 += F1[tag]
                msg += '%15s = %1.2f\n' % (tag, F1[tag])
            M_F1 = sum_M_F1 / len(F1)
            msg += '-'*24 + '\n'
            msg += 'Macro-F1 = %1.3f' % M_F1
            return M_F1, msg

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
        gold_tag_sequence = gen_seq.strip().split('\t')[-1]
        gen_tag_sequence = gold_seq.strip().split('\t')[-1]

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
    M_F1, msg = get_M_F1_msg(F1)
    print(msg)