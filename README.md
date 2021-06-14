# Argumentation Mining in Portuguese
This repository contains an End-to-End Sequence Tagging task for the Portuguese language.
It uses code from the following third-party repositories:
- Preprocessed data from [this repo](https://github.com/UKPLab/acl2017-neural_end2end_am);
- Annotation Projection algorithm from [this repo](https://github.com/UKPLab/coling2018-xling_argument_mining);
- [SimAlign](https://github.com/cisnlp/simalign) as the word alignment tool;
- Multilingual word embeddings from [this repo](https://github.com/facebookresearch/MUSE).

## Usage
Make use of the script files on [this folder](https://github.com/AfonsoSalgadoSousa/argumentation_mining_pt/tree/main/scripts) to build the annotation projection corpus or scripts on [this folder](https://github.com/AfonsoSalgadoSousa/argumentation_mining_pt/tree/main/tagger/scripts) for train and evaluation of the sequence tagging models. For sequence tagging, we adopt the model from NeuroNLP2. For a detailed explanation on how to use the tagger tool, please refer to the [NeuroNLP2 repository](https://github.com/XuezheMax/NeuroNLP2).

## Building the Portuguese Dataset
Assuming the following file structure:
  DATASET_ROOT_DIR
      ├── en_pe                   # Persuasive Essays ConLL-formatted
      |   └── train.dat            
      │   └── dev.dat
      │   └── test.dat

### Free-text
Firstly, create free-text files for each train/dev/test file.
```bash
python src/convert_to_free_text.py data/auxiliary/train_ft.txt data/en_pe/train.dat
python src/convert_to_free_text.py data/auxiliary/dev_ft.txt data/en_pe/dev.dat
python src/convert_to_free_text.py data/auxiliary/dev_ft.txt data/en_pe/test.dat
```
These scripts create the "auxiliary" folder to store the auxiliary files for the construction of the dataset.

### Translation
Next, translate the free-text files.

```bash
python src/translator.py data/auxiliary/train_ft.txt --src_lang en --trg_lang pt
python src/translator.py data/auxiliary/dev_ft.txt --src_lang en --trg_lang pt
python src/translator.py data/auxiliary/test_ft.txt --src_lang en --trg_lang pt
```
To execute the translation script, the following packages are required:
- NLTK
- PyTorch
- Transformers

### Alignment
Next, generate alignment files for the previously created files with translations.
```bash
python src/align.py data/auxiliary/train_ft_translated.txt
python src/align.py data/auxiliary/dev_ft_translated.txt
python src/align.py data/auxiliary/test_ft_translated.txt
```
To execute the alignment script, the following packages are required:
- NLTK
- NumPy
- SciPy
- PyTorch
- Scikit-learn
- Transformers 3.1.0 (later versions might throw an error)
- NetworkX
- tqdm

### Annotation Projection
Finally, project the annotations.
```bash
python src/project_annotations.py data/en_pe/train.dat data/auxiliary/train_ft_translated.txt data/auxiliary/train_ft_translated_alignment.txt --output_path data/pt_pe
python src/project_annotations.py data/en_pe/dev.dat data/auxiliary/dev_ft_translated.txt data/auxiliary/dev_ft_translated_alignment.txt --output_path data/pt_pe
python src/project_annotations.py data/en_pe/test.dat data/auxiliary/test_ft_translated.txt data/auxiliary/test_ft_translated_alignment.txt --output_path data/pt_pe
```
These scripts create the "pt_pe" folder to store the Portuguese version of the dataset.

To execute the annotation projection script, the following packages are required:
- NLTK

### Licenses