Cross-Lingual Annotation Projection for ArgumentMining in Portuguese
===============
This repository contains the code to replicate the experiments in the paper.
It uses code from the following third-party repositories:
- Preprocessed data from [this repo](https://github.com/UKPLab/acl2017-neural_end2end_am);
- [SimAlign](https://github.com/cisnlp/simalign) as the word alignment tool;
- Multilingual word embeddings from [this repo](https://github.com/facebookresearch/MUSE).

## Requirements
- NLTK
- NumPy
- SciPy
- PyTorch
- Scikit-learn
- Transformers 3.1.0 (later versions might throw an error)
- NetworkX
- tqdm

## Usage
Make use of the script files on [this folder](https://github.com/AfonsoSalgadoSousa/argumentation_mining_pt/tree/main/scripts) to build the annotation projection corpus and perform intrinsic evaluations (further explanations below). Alternatively, you can find the Portuguese dataset and all of the intermediate files in [this folder](https://github.com/AfonsoSalgadoSousa/argumentation_mining_pt/tree/main/data).

For sequence tagging, we adopted both TAGGER and NeuroNLP2. For a detailed explanation on how to use these tools, please refer to the [TAGGER](https://github.com/achernodub/targer) or [NeuroNLP2 repository](https://github.com/XuezheMax/NeuroNLP2).

## Building the Portuguese Dataset
To build the Portuguese version of the Persuasive Essays, we used the CoNLL-formatted version of the dataset, from [this repo](https://github.com/UKPLab/acl2017-neural_end2end_am). assuming the following file structure:
```angular2
  ├── DATASET_ROOT_DIR
  │   ├── en_pe                   # Persuasive Essays ConLL-formatted
  │   │   ├── train.dat            
  │   │   ├── dev.dat
  │   │   ├── test.dat
```

### Free-text
To start building the dataset, create free-text files for each train/dev/test file.
```bash
python src/convert_to_free_text.py data/auxiliary/train/train_ft.txt data/en_pe/train.dat
python src/convert_to_free_text.py data/auxiliary/dev/dev_ft.txt data/en_pe/dev.dat
python src/convert_to_free_text.py data/auxiliary/test/test_ft.txt data/en_pe/test.dat
```
These scripts create the "auxiliary" folder in the root folder to store further auxiliary files for the construction of the dataset.

### Translation
Next, translate the free-text files.

```bash
python src/translator.py data/auxiliary/train/train_ft.txt --src_lang en --trg_lang pt
python src/translator.py data/auxiliary/dev/dev_ft.txt --src_lang en --trg_lang pt
python src/translator.py data/auxiliary/test/test_ft.txt --src_lang en --trg_lang pt
```
You will end up with a file with parallel data seperated by the "|||" sequence, sentences split by a break line and paragraph split by an empty line.

### Alignment
Next, generate alignment files for the previously created files with translations.
```bash
python src/align.py data/auxiliary/train/train_ft_translated.txt
python src/align.py data/auxiliary/dev/dev_ft_translated.txt
python src/align.py data/auxiliary/test/test_ft_translated.txt
```
The generated file follows the structure from the translation file, but instead of parallel data has per-token index pairs. 

### Annotation Projection
Finally, project the annotations.
```bash
python src/project_annotations.py data/en_pe/train.dat data/auxiliary/train/train_ft_translated.txt data/auxiliary/train/train_ft_translated_alignment.txt --output_dir data/pt_pe
python src/project_annotations.py data/en_pe/dev.dat data/auxiliary/dev/dev_ft_translated.txt data/auxiliary/dev/dev_ft_translated_alignment.txt --output_dir data/pt_pe
python src/project_annotations.py data/en_pe/test.dat data/auxiliary/test/test_ft_translated.txt data/auxiliary/test/test_ft_translated_alignment.txt --output_dir data/pt_pe
```
These scripts create the "pt_pe" folder to store the Portuguese version of the dataset.

## Evaluation
We performed both intrinsic and extrinsic evaluation of the corpus.

### Intrinsic Evaluation
To replicate the results in the paper for intrinsic evaluation, run:
```bash
python src/eval_en_from_pt.py data/en_pe/ data/en_from_pt_pe/
```
Alternatively, you can run the same evaluation per split using the _--split_ tag, and with or without padding with _--with_pad_.

### Licenses
There are two licenses for this project:

- The [first one](https://github.com/AfonsoSalgadoSousa/argumentation_mining_pt/tree/main/LICENSE) applies to all files except for the [data folder](https://github.com/AfonsoSalgadoSousa/argumentation_mining_pt/tree/main/data);
- The other license applies to the [data folder](https://github.com/AfonsoSalgadoSousa/argumentation_mining_pt/tree/main/data). This folder contains both the [PE corpus in English](https://github.com/UKPLab/acl2017-neural_end2end_am/tree/master/data/conllm) (data/en_pe), as well as the subsequent translated, aligned, and projected Portuguese files. The license can be found [here](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422).
