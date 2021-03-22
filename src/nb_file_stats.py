# %%
from pathlib import Path

import nltk.data
from nltk import word_tokenize

split = 'train'
# Load data
nltk_text = nltk.data.load(f'../data/auxiliary/{split}_ft.txt')

print('# tokens: {}'.format(len(nltk_text)))

tokens = word_tokenize(nltk_text)
print('Vocab length: {}'.format(len(tokens)))

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = tokenizer.tokenize(nltk_text)
print('# sentences: {}'.format(len(sentences)))

paragraphs = nltk_text.split("\t")
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
