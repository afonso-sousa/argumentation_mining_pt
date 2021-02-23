#!/bin/sh
python src/project_annotations.py data/en_pe/train.dat data/auxiliary/train_ft_translated.txt data/auxiliary/train_ft_translated_alignment.txt --output_path data/pt_pe
