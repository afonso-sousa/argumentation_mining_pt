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


def detect_bios(labels):
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


def translation_indices(indices, alignment):
    alignment_dict = {}
    for align in alignment.split():
        src_idx, trg_idx = list(map(int, align.split("-")))
        if src_idx in alignment_dict:
            alignment_dict[src_idx] = alignment_dict[src_idx]+[trg_idx]
        else:
            alignment_dict[src_idx] = [trg_idx]

    aligns = []
    for start, end, component_type in indices:
        trg_align_list = []
        for idx in range(start, end+1):
            if idx in alignment_dict:
                trg_align_list.extend(alignment_dict[idx])
        trg_align_list.sort()

        if not trg_align_list:
            break
        idx_start = trg_align_list[0]
        if idx_start - 1 == min(alignment_dict):
            idx_start -= 1
        idx_end = trg_align_list[-1]
        if len(aligns) > 0:
            prev_end_idx = aligns[-1][1]
            prev_start_idx = aligns[-1][0]
            if idx_start <= prev_end_idx:
                idx_start = prev_end_idx+1

        aligns.append((idx_start, idx_end, component_type))
    return aligns


def printout(idx, sequence, output_path, component_type="O"):
    tmp = idx
    with open(output_path, 'a+') as output_file:
        for itoken, token in enumerate(sequence):
            if component_type != "O":
                if itoken == 0:
                    pre = "B-"
                else:
                    pre = "I-"
            else:
                pre = ""
            if not component_type:
                component_type = "O"
            output_file.write(f'{tmp}\t{token}\t{pre}{component_type}\n')
            tmp += 1
    return tmp


def process(sentences, sentences_alignments, labels, fout):
    last = 0
    idx = 1
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

        # src_idx = list(zip(*tuple_align))[0]
        # trg_idx = list(zip(*tuple_align))[1]
        # len_trg_sent = len(trg_tokens)
        # for start, end, component_type in indices:
        #     for idx in range(min(src_idx), max(trg_idx) + 1):
        #         if idx not in trg_idx:
        #             print(trg_tokens[idx])

            # aligned_not_used = [idx for idx in trg_idx if (start <= idx <= end + 1) and idx not in range(start, end+1)]
            # print([trg_tokens[idx] for idx in aligned_not_used])

        last = last+len_src_sent
        aligns = sorted(translation_indices(indices, align))
        prev = 0
        for start, end, component_type in aligns:
            if start > end:
                continue
            before_adu = trg_tokens[prev:start]
            adu = trg_tokens[start:end+1]
            if before_adu != []:
                idx = printout(idx, before_adu, fout)
            idx = printout(idx, adu, fout, component_type)
            prev = end+1
        after_adu = trg_tokens[prev:]
        if after_adu != []:
            idx = printout(idx, after_adu, fout)


def create_conll(corpus_path, alignments, translations, output_path):
    annotation_dict = read_doc(corpus_path)
    for paragraph, labels in annotation_dict.items():
        sentences = []
        sentences_alignments = []
        sentences_in_paragraph = sent_tokenize(paragraph)
        seen_sentences = []
        for idx, trans in enumerate(translations):
            src, trg = trans.split(" ||| ")
            if src.strip() in sentences_in_paragraph and not src.strip() in seen_sentences:
                sentences.append((src, trg))
                sentences_alignments.append(alignments[idx])
                seen_sentences.append(src.strip())
        process(sentences, sentences_alignments, labels, output_path)
        with open(output_path, 'a+') as output_file:
            output_file.write("\n")


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project annotations and create translated ConLL-formatted train/dev/test split.",
                                     epilog="example: python project_annotations.py train_conll train_translated_file train_alignment_file")
    parser.add_argument("corpus_path", type=Path)
    parser.add_argument("translation_path", type=Path)
    parser.add_argument("alignment_path", type=Path)
    parser.add_argument("--output_path", type=Path, default=".")
    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)

    alignments = open(args.alignment_path).readlines()
    translations = open(args.translation_path).readlines()

    create_conll(args.corpus_path, alignments,
                 translations, args.output_path / f'{args.corpus_path.stem}_pt.dat')

# %%
# translations = open("../data/auxiliary/train_ft_translated.txt").readlines()
# alignments = open(
#     "../data/auxiliary/train_ft_translated_alignment.txt").readlines()
# create_conll("../data/en_pe/train.dat",
#              alignments, translations, "../TO_REMOVE.dat")
