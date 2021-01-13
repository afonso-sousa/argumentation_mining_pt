#!/bin/sh
python src/project_annotations.py data/persuasive_essays/Paragraph_Level/train.dat.abs data/persuasive_essays/Paragraph_Level/test.dat.abs data/persuasive_essays/Paragraph_Level/dev.dat.abs data/auxiliary/all_ft_translated.txt data/auxiliary/all_ft_translated_alignment.txt --output_path data/pt_pe
