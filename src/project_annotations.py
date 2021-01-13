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


K = 1


def isConsecutive(lst, descending=False):
    last = None
    for x in lst:
        if last is not None:
            next_val = last-1 if descending else last+1
            if x != next_val:
                return False
        last = x
    return True


def findExtremeConsecutive(lst, reverse=True, k=1):
    s = sorted(lst, reverse=reverse)
    for ix, x in enumerate(s):
        mylst = s[ix:ix+k]
        if isConsecutive(mylst, descending=reverse):
            return x
    return s[0]


def detect_bios(labels):
    indices = []
    startComponent = False
    startindex = 0
    component_type = 'O'
    for index, tok in enumerate(labels):
        _, token = tok
        if startComponent == True and token.startswith("B-"):
            endindex = index-1
            indices.append((startindex, endindex, component_type))
            startindex = index
            component_type = token.split(":")[0][2:]
            startComponent = True
        elif startComponent == True and token.startswith("O"):
            endindex = index-1
            indices.append((startindex, endindex, component_type))
            startComponent = False
        elif token.startswith("B-"):
            component_type = token.split(":")[0][2:]
            startComponent = True
            startindex = index
        elif token.startswith("I-"):
            endindex = index
            indices.append((startindex, endindex, component_type))
    return indices


def getTranslationIndices(indices, align):
    h = {}
    for y in align.split():
        a, b = list(map(int, y.split("-")))
        if a in h:
            h[a] = h[a]+[b]
        else:
            h[a] = [b]
    aligns = []
    for x in indices:
        start, end, component_type = x
        q = []
        for z in range(start, end+1):
            q.append(h.get(z))
        qq = list(filter(lambda x: x != None, q))
        flat_list = [item for sublist in qq for item in sublist]
        if not flat_list:
            break
        for myK in range(K, 0, -1):
            indexStart, indexEnd = findExtremeConsecutive(
                flat_list, reverse=False, k=K), findExtremeConsecutive(flat_list, reverse=True, k=myK)
            if len(aligns) > 0:
                indexEndPrev = aligns[-1][1]
                indexStartPrev = aligns[-1][0]
                if indexStart <= indexEndPrev:
                    sys.stderr.write("DOESN'T WORK OUT %d %d\n" %
                                     (indexStart, indexEndPrev))
                    if indexEnd < indexStartPrev:
                        sys.stderr.write("Li'l non-monotonity\n")
                        break
                    indexStart = indexEndPrev+1
            if indexStart <= indexEnd:
                break
        if indexStart > indexEnd:
            sys.stderr.write(str(aligns))
            sys.stderr.write("ERROR SOMEWHERE: %d %d\n" %
                             (indexStart, indexEnd))
        aligns.append((indexStart, indexEnd, component_type))
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
        m = len(src_tokens)
        align = sentences_alignments[i].strip()
        curLabels = labels[last:last+m]
        indices = detect_bios(curLabels)
        last = last+m
        aligns = sorted(getTranslationIndices(indices, align))
        prev = 0
        for start, end, component_type in aligns:
            if start > end:
                continue
            before = trg_tokens[prev:start]
            middle = trg_tokens[start:end+1]
            if before != []:
                idx = printout(idx, before, fout)
            idx = printout(idx, middle, fout, component_type)
            prev = end+1
        after = trg_tokens[prev:]
        if after != []:
            idx = printout(idx, after, fout)


def create_conll(corpus_path, alignments, translations, output_path):
    _, annotation_dict = read_doc(corpus_path)
    for paragraph, labels in annotation_dict.items():
        sentences = []
        sentences_alignments = []
        sentences_in_paragraph = sent_tokenize(paragraph)
        seen_sentences = []
        for idx, trans in enumerate(translations):
            src, trg = trans.split(" ||| ")
            if src.strip() in sentences_in_paragraph and not src.strip() in seen_sentences:
                # print(src)
                sentences.append((src, trg))
                sentences_alignments.append(alignments[idx])
                seen_sentences.append(src.strip())
        process(sentences, sentences_alignments, labels, output_path)
        with open(output_path, 'a+') as output_file:
            output_file.write("\n")
        # sys.exit(1)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project annotations and create translated ConLL-formatted train/dev/test split.",
                                     epilog="example: python project_annotations.py train_conll test_conll dev_conll translated_file alignment_file")
    parser.add_argument("train_corpus_path", type=Path)
    parser.add_argument("test_corpus_path", type=Path)
    parser.add_argument("dev_corpus_path", type=Path)
    parser.add_argument("translation_path", type=Path)
    parser.add_argument("alignment_path", type=Path)
    parser.add_argument("--output_path", type=Path, default=".")
    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)

    alignments = open(args.alignment_path).readlines()
    translations = open(args.translation_path).readlines()

    create_conll(args.train_corpus_path, alignments,
                 translations, args.output_path / "pt_train.dat")
    create_conll(args.test_corpus_path, alignments,
                 translations, args.output_path / "pt_test.dat")
    create_conll(args.dev_corpus_path, alignments,
                 translations, args.output_path / "pt_dev.dat")


# %%
# translations = open("data/auxiliary/all_ft_translated.txt").readlines()
# alignments = open("data/auxiliary/all_ft_translated_alignment.txt").readlines()
# create_conll("data/persuasive_essays/Paragraph_Level/train.dat.abs",
#              alignments, translations, "data/pt_pe/pt_train.dat")

# %%
