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
    rel_idx = '-1'
    rel_type = 'none'

    for index, tok in enumerate(labels):
        _, token = tok
        # contiguous component started
        if start_component == True and token.startswith("B-"):
            endindex = index-1
            indices.append((startindex, endindex, component_type, rel_idx, rel_type))
            startindex = index
            all_cols = token.split(":")
            component_type = all_cols[0][2:]
            if len(all_cols) == 1:
                rel_idx = '-1'
                rel_type = 'none'
            elif len(all_cols) == 2:
                rel_idx = '-1'
                rel_type = all_cols[1]
            elif len(all_cols) == 3:
                rel_idx = all_cols[1]
                rel_type = all_cols[2]
            start_component = True
        # component finished
        elif start_component == True and token.startswith("O"):
            endindex = index-1
            indices.append((startindex, endindex, component_type, rel_idx, rel_type))
            start_component = False
        # component started
        elif token.startswith("B-"):
            all_cols = token.split(":")
            component_type = all_cols[0][2:]
            if len(all_cols) == 1:
                rel_idx = '-1'
                rel_type = 'none'
            elif len(all_cols) == 2:
                rel_idx = '-1'
                rel_type = all_cols[1]
            elif len(all_cols) == 3:
                rel_idx = all_cols[1]
                rel_type = all_cols[2]
            start_component = True
            startindex = index
    if token.startswith("I-"):
        endindex = index
        indices.append((startindex, endindex, component_type, rel_idx, rel_type))
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
    for start, end, component_type, rel_idx, rel_type in indices:
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

        aligns.append((idx_start, idx_end, component_type, rel_idx, rel_type))
    return aligns, count


def printout(idx, sequence, output_path, component_type="O", rel_idx="-1", rel_type="none"):
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
            if (rel_idx == "-1" and rel_type != "none"):
                output_file.write(f'{tmp}\t{token}\t{pre}{component_type}{":"}{rel_type}\n')
            elif (rel_idx != "-1" and rel_type != "none"):
                output_file.write(f'{tmp}\t{token}\t{pre}{component_type}{":"}{rel_idx}{":"}{rel_type}\n')
            else:
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
        for start, end, component_type, rel_idx, rel_type in aligns:
            if start == 78:
                print('-----', start, end, component_type)
                sys.exit(1)
            if start >= end:
                continue
            before_adu = trg_tokens[prev:start]
            if before_adu != []:
                idx = printout(idx, before_adu, fout)

            adu = trg_tokens[start:end+1]
            idx = printout(idx, adu, fout, component_type, rel_idx, rel_type)

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

def create_conll_hardcode(corpus_path, alignments, translations, output_path, pad_verbosity=True, reverse=False):

    annotation_dict = [
        ('To sum up , technology has helped us to have more comfortable life .', 
        [
        ('To', 'O'),
        ('sum', 'O'), 
        ('up', 'O'), 
        (',', 'O'), 
        ('technology', 'B-MajorClaim'), 
        ('has', 'I-MajorClaim'),
        ('helped', 'I-MajorClaim'), 
        ('us', 'I-MajorClaim'), 
        ('to', 'I-MajorClaim'), 
        ('have', 'I-MajorClaim'),
        ('more', 'I-MajorClaim'), 
        ('confortable', 'I-MajorClaim'), 
        ('life', 'I-MajorClaim'), 
        ('.', 'O')
        ]),

        ('The last 50 years have seen a significant increase in the number of tourist traveling worldwide . While some might think the tourism bring large profit for the destination countries , I would contend that this industry has affected the cultural attributes and damaged the natural environment of the tourist destinations .', 
        [
        ('The', 'O'),
        ('last', 'O'), 
        ('50', 'O'), 
        ('years', 'O'), 
        ('have', 'O'), 
        ('seen', 'O'),
        ('a', 'O'), 
        ('significant', 'O'), 
        ('increase', 'O'), 
        ('in', 'O'),
        ('the', 'O'), 
        ('number', 'O'), 
        ('of', 'O'), 
        ('tourist', 'O'),
        ('traveling', 'O'),
        ('worldwide', 'O'),
        ('.', 'O'),
        ('While', 'O'),
        ('some', 'O'), 
        ('might', 'O'), 
        ('think', 'O'), 
        ('the', 'B-Claim'), 
        ('tourism', 'I-Claim'),
        ('bring', 'I-Claim'), 
        ('large', 'I-Claim'), 
        ('profit', 'I-Claim'), 
        ('for', 'I-Claim'),
        ('the', 'I-Claim'), 
        ('destination', 'I-Claim'), 
        ('countries', 'I-Claim'), 
        (',', 'O'),
        ('I', 'O'),
        ('would', 'O'),
        ('contend', 'O'),
        ('that', 'O'),
        ('this', 'B-MajorClaim:6:Against'),
        ('industry', 'I-MajorClaim:6:Against'),
        ('has', 'I-MajorClaim:6:Against'),
        ('affected', 'I-MajorClaim:6:Against'),
        ('the', 'I-MajorClaim:6:Against'),
        ('cultural', 'I-MajorClaim:6:Against'),
        ('attributes', 'I-MajorClaim:6:Against'),
        ('and', 'I-MajorClaim:6:Against'),
        ('damaged', 'I-MajorClaim:6:Against'),
        ('the', 'I-MajorClaim:6:Against'),
        ('natural', 'I-MajorClaim:6:Against'),
        ('environment', 'I-MajorClaim:6:Against'),
        ('of', 'I-MajorClaim:6:Against'),
        ('the', 'I-MajorClaim:6:Against'),
        ('tourist', 'I-MajorClaim:6:Against'),
        ('destinations', 'I-MajorClaim:6:Against'),
        ('.', 'O')
        ])

        ]

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

        c = process(sentences, sentences_alignments, labels, output_path, pad_verbosity)
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

    create_conll(args.corpus_path, alignments, translations, output_path, args.pad_verbosity, args.reverse)

"""     translations = [
    'To sum up , technology has helped us to have more comfortable life . ||| Resumindo , a tecnologia nos ajudou a ter uma vida mais confotável .', 
    '\n', 
    'The last 50 years have seen a significant increase in the number of tourist traveling worldwide . ||| Os últimos 50 anos têm visto um aumento significativo no número de turistas viajando em todo o mundo . ',
    'While some might think the tourism bring large profit for the destination countries , I would contend that this industry has affected the cultural attributes and damaged the natural environment of the tourist destinations . ||| Embora alguns possam pensar que o turismo traz grande lucro para os países de destino , eu afirmaria que esta indústria tem afetado os atributos culturais e danificado o ambiente natural dos destinos turísticos . ',
    '\n']

    alignments = [
    '0-2 1-0 2-0 3-1 4-3 5-5 6-5 7-4 8-6 9-7 10-10 11-11 12-9 13-12', 
    '\n', 
    '0-0 1-1 2-2 3-3 4-4 5-5 6-6 7-8 8-7 9-9 11-10 12-11 13-12 14-13 15-17 16-18',
    '0-0 1-1 2-2 3-3 4-5 5-6 6-7 7-8 8-9 9-10 10-11 11-14 12-12 13-15 14-16 15-17 16-17 17-18 18-19 19-20 20-21 21-22 22-23 23-25 24-24 25-26 26-27 27-28 28-30 29-29 30-31 32-33 33-32 34-34',
    '\n']

    create_conll_hardcode("C:/Users/Bernardo/Desktop/argumentation_mining_pt/data/en_pe/small_test.dat", alignments, translations, "../TO_REMOVE.dat", False, False)
 """

# %%
# Single example test
"""
# replace line 191 with this to test
annotation_dict = [('To sum up , technology has helped us to have more comfortable life .', [('To', 'O'),
    ('sum', 'O'), ('up', 'O'), (',', 'O'), ('technology', 'B-MajorClaim'), ('has', 'I-MajorClaim'),
    ('helped', 'I-MajorClaim'), ('us', 'I-MajorClaim'), ('to', 'I-MajorClaim'), ('have', 'I-MajorClaim'),
    ('more', 'I-MajorClaim'), ('confortable', 'I-MajorClaim'), ('life', 'I-MajorClaim'), ('.', 'O')])]
translations = ['To sum up , technology has helped us to have more comfortable life . ||| Resumindo , a tecnologia nos ajudou a ter uma vida mais confotável .', '\n']
alignments = ['0-2 1-0 2-0 3-1 4-3 5-5 6-5 7-4 8-6 9-7 10-10 11-11 12-9 13-12', '\n']
create_conll("../data/en_pe/dev.dat",
             alignments, translations, "../TO_REMOVE.dat", True, False)
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
