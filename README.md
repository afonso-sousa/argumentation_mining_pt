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

## Usage
Make use of the script files on [this folder](https://github.com/AfonsoSalgadoSousa/argumentation_mining_pt/tree/main/scripts) to build the annotation projection corpus or scripts on [this folder](https://github.com/AfonsoSalgadoSousa/argumentation_mining_pt/tree/main/tagger/scripts) for train and evaluation of the sequence tagging models.
For a detail explanation on how to use the tagger tool to train/evaluate/save models, please refer to the [Tagger repo](https://github.com/achernodub/targer).
