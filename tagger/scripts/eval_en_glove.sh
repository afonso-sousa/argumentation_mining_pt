cd tagger
python run_tagger.py trained/glove_en_tagger.hdf5 ../data/persuasive_essays/Paragraph_Level/test.dat.abs -d connl-pe -v f1-macro
