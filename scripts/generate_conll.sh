#!/bin/sh
python src/project_annotations.py data/en_pe/train.dat data/auxiliary/train_ft_translated.txt data/auxiliary/train_ft_translated_alignment.txt --output_dir data/pt_pe
# python src/project_annotations.py data/pt_pe/train.dat data/auxiliary/train_ft_translated.txt data/auxiliary/train_ft_translated_alignment_src-pt.txt --output_dir data/en_from_pt_pe --pad_verbosity --reverse

