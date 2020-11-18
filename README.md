# Argumentation Mining in Portuguese
This repository contains an End-to-End Sequence Tagging task for the Portuguese language.
It uses code from the following third-party repositories:
- Preprocessed data from [this repo](https://github.com/UKPLab/acl2017-neural_end2end_am);
- Annotation Projection algorithm from [this repo](https://github.com/UKPLab/coling2018-xling_argument_mining);
- [SimAlign](https://github.com/cisnlp/simalign) as the word alignment tool;
- [Targer](https://github.com/achernodub/targer) as a neural tagger;
- Multilingual word embeddings from [this repo](https://github.com/facebookresearch/MUSE).

## Requirements

- Python 3.5.2 or higher
- NumPy 1.15.1
- SciPy 1.1.0
- PyTorch >= 0.4.1

## Evaluation

```python
python run_tagger.py 2020_11_18_15-15_51_tagger.hdf5 ../data/AM/persuasive_essays/Paragraph_Level/test.dat.abs -d connl-pe
```

## Usage

### Train/test

The commands I use to train my model on English and Portuguese embeddings were:
```python
cd tagger
python3 main.py --train ../data/AM/persuasive_essays/Paragraph_Level/train.dat.abs --dev ../data/AM/persuasive_essays/Paragraph_Level/dev.dat.abs --test ../data/AM/persuasive_essays/Paragraph_Level/test.dat.abs --data-io connl-pe --evaluator f1-alpha-match-10 --opt adam --lr 0.001 --save-best yes --patience 20 --rnn-hidden-dim 200 --epoch-num 2 --emb-fn embeddings/wiki.multi.en.vec --emb-dim 300
```
For a detail explanation on how to use the tagger tool to train/evaluate/save models, please refer to the [Tagger repo](https://github.com/achernodub/targer).
