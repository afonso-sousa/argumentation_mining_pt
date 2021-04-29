# %%
from pathlib import Path

import nltk.data
from nltk import word_tokenize


split = 'test'
gold_standard_dir = Path("../data/pt_pe_pad")

if split == 'all':
    gold_train = open(gold_standard_dir / 'train.dat').readlines()
    gold_dev = open(gold_standard_dir / 'dev.dat').readlines()
    gold_test = open(gold_standard_dir / 'test.dat').readlines()
    gold_standard = []
    for split in [gold_train, gold_dev, gold_test]:
        gold_standard.extend(split)
else:
    gold_standard = open(gold_standard_dir / f'{split}.dat').readlines()

tokens = []
for gold_seq in gold_standard:
        if gold_seq == '\n':
            continue
        tokens.append(gold_seq.strip().split('\t')[1])

print('# tokens: {}'.format(len(tokens)))

print('Vocab length: {}'.format(len(set(tokens))))

# %%
nltk_text = nltk.data.load(f'../data/auxiliary/{split}/{split}_ft.txt')
tokens = word_tokenize(nltk_text)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = tokenizer.tokenize(nltk_text)
print('# sentences: {}'.format(len(sentences)))

paragraphs = nltk_text.split("\t")
print('# paragraphs: {}'.format(len(paragraphs)))

# %%
import utils
paragraphs = utils.read_doc(f"../data/pt_pe/{split}.dat")
text = "\t".join(list(zip(*paragraphs))[0])

print('# tokens: {}'.format(len(text)))

tokens = word_tokenize(text)
print('Vocab length: {}'.format(len(tokens)))

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = tokenizer.tokenize(text)
print('# sentences: {}'.format(len(sentences)))

paragraphs = text.split("\t")
print('# paragraphs: {}'.format(len(paragraphs)))

# %%
translation = open(f'../data/auxiliary/{split}_ft_translated.txt').readlines()
print(f'Translation sentences: {len(translation)}')

# %%
alignments = open(f'../data/auxiliary/{split}_ft_translated_alignment.txt').readlines()
print(f'Alignments sentences: {len(alignments)}')

# %%
conll = open(f'../data/pt_pe/{split}_pt.dat').readlines()
print(f'Conll tokens: {len(conll)}')
paragraphs = 0
for line in conll:
    if not line.strip():
        paragraphs += 1
print('# paragraphs: {}'.format(paragraphs))

# %%
gold_standard_dir = Path("../data/en_pe")
gold_train = open(gold_standard_dir / 'train.dat').readlines()
gold_dev = open(gold_standard_dir / 'dev.dat').readlines()
gold_test = open(gold_standard_dir / 'test.dat').readlines()
gold_standard = []
for split in [gold_train, gold_dev, gold_test]:
    gold_standard.extend(split)

# %%
tokens = []
for gold_seq in gold_standard:
        if gold_seq == '\n':
            continue
        tokens.append(gold_seq.strip().split('\t')[1])


# %%
print('# tokens: {}'.format(len(gold_standard)))

tokens = word_tokenize(gold_standard)
print('Vocab length: {}'.format(len(tokens)))

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = tokenizer.tokenize(gold_standard)
print('# sentences: {}'.format(len(sentences)))

paragraphs = gold_standard.split("\t")
print('# paragraphs: {}'.format(len(paragraphs)))
# %%
