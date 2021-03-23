"""
Script base from https://github.com/UKPLab/coling2018-xling_argument_mining/blob/master/code/annotationProjection/projectArguments.py
"""
# %%

import argparse
import os
import sys
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize

from utils import read_doc

# nltk.download('punkt')


def free_text_to_sentences(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return tokenizer.tokenize(text)


def detect_bios(labels):
    if labels == []:
        return []
    indices = []
    start_component = False
    startindex = 0
    component_type = 'O'
    for index, tok in enumerate(labels):
        _, token = tok
        # contiguous component started
        if start_component == True and token.startswith("B-"):
            endindex = index-1
            indices.append((startindex, endindex, component_type))
            startindex = index
            component_type = token.split(":")[0][2:]
            start_component = True
        # component finished
        elif start_component == True and token.startswith("O"):
            endindex = index-1
            indices.append((startindex, endindex, component_type))
            start_component = False
        # component started
        elif token.startswith("B-"):
            component_type = token.split(":")[0][2:]
            start_component = True
            startindex = index
    if token.startswith("I-"):
        endindex = index
        indices.append((startindex, endindex, component_type))
    return indices


def pad_verbosity(value, trg_aligns, right=False):
    new_value = value
    val_idx = trg_aligns.index(value)
    if right:
        aligns_subset = trg_aligns[val_idx:]
        integer_list = range(value, max(aligns_subset))
    else:
        aligns_subset = trg_aligns[:val_idx]
        integer_list = range(value-1, -1, -1)
    for i in integer_list:
        if i in aligns_subset:
            break
        if right:
            new_value += 1
        else:
            new_value -= 1

    count_simple = 0
    count_multi = 0
    if new_value != value:
        print(f'Val: {value} -> New Val: {new_value}')
        count_simple = 1
        if new_value - value >= 2:
            count_multi = 1
    print('OK')
    return new_value, (count_simple, count_multi)


def translation_indices(indices, alignment, pad_verb):
    alignment_dict = {}
    for align in alignment.split():
        src_idx, trg_idx = list(map(int, align.split("-")))
        if src_idx in alignment_dict:
            alignment_dict[src_idx] = alignment_dict[src_idx]+[trg_idx]
        else:
            alignment_dict[src_idx] = [trg_idx]

    aligns = []
    count = (0, 0, len(indices))
    for start, end, component_type in indices:
        trg_align_list = []
        for idx in range(start, end+1):
            if idx in alignment_dict:
                trg_align_list.extend(alignment_dict[idx])
        trg_align_list.sort()

        if not trg_align_list:
            break
        idx_start = trg_align_list[0]
        idx_end = trg_align_list[-1]

        # pad portuguese verbosity
        if pad_verb:
            all_trg_align_list = list(alignment_dict.values())
            all_trg_align_list = [
                item for sublist in all_trg_align_list for item in sublist]
            idx_start, c = pad_verbosity(idx_start, sorted(all_trg_align_list))
            count = (count[0] + c[0], count[1] + c[1], count[2])

        # make ADUs disjoint if alignments are wrong
        if len(aligns) > 0:
            prev_end_idx = aligns[-1][1]
            if idx_start <= prev_end_idx:
                idx_start = prev_end_idx+1

        aligns.append((idx_start, idx_end, component_type))
    return aligns, count


def printout(idx, sequence, output_path, component_type="O"):
    tmp = idx
    with open(output_path, 'a+') as output_file:
        for i, token in enumerate(sequence):
            if component_type != "O":
                if i == 0:
                    pre = "B-"
                else:
                    pre = "I-"
            else:
                pre = ""
            output_file.write(f'{tmp}\t{token}\t{pre}{component_type}\n')
            tmp += 1
    return tmp


def process(sentences, sentences_alignments, labels, fout, pad_verbosity):
    last = 0
    idx = 1
    count = (0, 0, 0)
    for i in range(len(sentences)):
        src, trg = sentences[i]
        src_tokens = src.split()
        trg_tokens = trg.split()
        len_src_sent = len(src_tokens)
        align = sentences_alignments[i].strip()
        curr_labels = labels[last:last+len_src_sent]
        indices = detect_bios(curr_labels)

        # convert alignments string to list of tuples
        # '0-0 1-1 2-2' -> [(0, 0), (1, 1), (2, 2)]
        # tuple_align = list(
        #     map(lambda x: tuple(map(int, x.split("-"))), align.split()))

        last = last+len_src_sent
        align_tuples, c = translation_indices(indices, align, pad_verbosity)
        aligns = sorted(align_tuples)
        count = (count[0] + c[0], count[1] + c[1], count[2] + c[2])
        prev = 0
        for start, end, component_type in aligns:
            if start == 78:
                print('-----', start, end, component_type)
                sys.exit(1)
            if start >= end:
                continue
            before_adu = trg_tokens[prev:start]
            if before_adu != []:
                idx = printout(idx, before_adu, fout)

            adu = trg_tokens[start:end+1]
            idx = printout(idx, adu, fout, component_type)

            prev = end+1
        after_adu = trg_tokens[prev:]
        if after_adu != []:
            idx = printout(idx, after_adu, fout)
    return count


def count_sent_translation_parag(translations, position):
    for i, line in enumerate(translations[position:]):
        if line == '\n':
            return i


def create_conll(corpus_path, alignments, translations, output_path, pad_verbosity=True, reverse=False):
    annotation_dict = read_doc(corpus_path)
    count = (0, 0, 0)
    total_sent = 0
    curr_idx = 0

    for paragraph, labels in annotation_dict:
        sentences_alignments = []
        # sentences_in_paragraph = free_text_to_sentences(paragraph)
        num_sentences = count_sent_translation_parag(translations, curr_idx)
        print(f'Index range: {curr_idx} - {curr_idx+num_sentences}')
        print(f'Num sentences: {num_sentences}')
        if not reverse:
            sentences = [tuple(trans.split(" ||| "))
                         for trans in translations[curr_idx:curr_idx+num_sentences]]
        else:
            sentences = [tuple(trans.split(" ||| "))[::-1]
                         for trans in translations[curr_idx:curr_idx+num_sentences]]
        sentences_alignments = alignments[curr_idx:curr_idx+num_sentences]

        """
        for s in list(zip(*sentences))[0]:
            if s.strip() not in sentences_in_paragraph:
                print(s)
                print(sentences_in_paragraph)
                sys.exit(1)
        """

        if sentences == []:
            print(paragraph)
            sys.exit(1)

        """
        with open("testing.txt", 'a+') as test:
            for s in list(zip(*sentences))[1]:
                test.write(s)
                test.write("\n")
            test.write("\n")
        """

        c = process(sentences, sentences_alignments,
                    labels, output_path, pad_verbosity)
        count = (count[0] + c[0], count[1] + c[1], count[2] + c[2])
        total_sent += num_sentences
        curr_idx += num_sentences + 1
        with open(output_path, 'a+') as output_file:
            output_file.write("\n")
    if pad_verbosity:
        print(f'Single: {count[0]}, Multi: {count[1]}, Total: {count[2]}')
    print(f'# sentences: {total_sent}')


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project annotations and create translated ConLL-formatted train/dev/test split.",
                                     epilog="example: python project_annotations.py train_conll train_translated_file train_alignment_file")
    parser.add_argument("corpus_path", type=Path)
    parser.add_argument("translation_path", type=Path)
    parser.add_argument("alignment_path", type=Path)
    parser.add_argument("--output_dir", type=Path, default=".")
    parser.add_argument('--pad_verbosity', action='store_true', default=False)
    parser.add_argument('--reverse', action='store_true', default=False)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    alignments = open(args.alignment_path).readlines()
    translations = open(args.translation_path).readlines()

    print(f'Corpus path: {args.corpus_path}')
    print(f'Length alignments: {len(alignments)}')
    print(f'Length translations: {len(translations)}')
    output_path = args.output_dir / f'{args.corpus_path.stem}.dat'
    print(f'Output path: {output_path}')
    print(f'Pad Verbosity: {args.pad_verbosity}')
    print(f'Reverse: {args.reverse}')

    create_conll(args.corpus_path, alignments,
                 translations, output_path, args.pad_verbosity, args.reverse)

# %%
# Single example test
"""
annotation_dict = {'In addition , sometimes animals from hot countries have to survive in the cold winter of somewhere in Europe .': [('In', 'O'), ('addition', 'O'), (',', 'O'), ('sometimes', 'B-Premise'), ('animals', 'I-Premise'), ('from', 'I-Premise'), ('hot', 'I-Premise'), ('countries', 'I-Premise'), ('have', 'I-Premise'), ('to', 'I-Premise'), ('survive', 'I-Premise'), ('in', 'I-Premise'), ('the', 'I-Premise'), ('cold', 'I-Premise'), ('winter', 'I-Premise'), ('of', 'I-Premise'), ('somewhere', 'I-Premise'), ('in', 'I-Premise'), ('Europe', 'I-Premise'), ('.', 'O')]}
translations = ['In addition , sometimes animals from hot countries have to survive in the cold winter of somewhere in Europe . ||| Além disso , por vezes , animais de países quentes têm de sobreviver no frio inverno de algum lugar na Europa .']
alignments = ['1-1 2-2 3-4 4-6 5-7 6-9 7-8 8-10 9-11 10-12 11-13 13-14 14-15 15-16 16-17 16-18 17-19 18-20 19-21']
create_conll("../data/en_pe/dev.dat",
             alignments, translations, "../TO_REMOVE.dat")
"""
# %%
# Creation test
"""
split = 'train'
pad_verb = True
reverse = False
translations = open(f'../data/auxiliary/{split}/{split}_ft_translated.txt').readlines()
alignments = open(
    f'../data/auxiliary/{split}/{split}_ft_translated_alignment{"_src-pt" if reverse else ""}.txt').readlines()
create_conll(f'../data/{"pt_pe" if reverse else "en_pe"}/{split}.dat', alignments,
                 translations, "../TO_REMOVE.dat", pad_verb, reverse)
"""

# %%
"""
translations = open("../data/auxiliary/train_ft_translated.txt").readlines()
alignments = open(
    "../data/auxiliary/train_ft_translated_alignment_src-pt.txt").readlines()
create_conll("../data/pt_pe/train_pt.dat", alignments,
                 translations, "../TO_REMOVE.dat", pad_verbosity=True, reverse=True)
"""
# %%
"""
gen_corpus = open('../TO_REMOVE.dat').readlines()
gold_standard = open(f'../data/en_pe/{split}.dat').readlines()

print(f'Length Generated Corpus: {len(gen_corpus)}')
print(f'Length Gold Standard: {len(gold_standard)}')
"""

# %%
"""
correct = 0
incorrect = 0
total = 0
gold_idx = 0
for idx, row in enumerate(gold_standard):
    total += 1
    gold = row.strip().split('\t')
    gen = gen_corpus[gold_idx].strip().split('\t')

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
    gold_idx += 1
print(f'Correct: {correct}')
print(f'Incorrect: {incorrect}')
print(f'Total: {total}')
"""

# %%
