cd tagger
python run_tagger.py trained/glove_en_200_epoch_tagger.hdf5 ../data/en_pe/test.dat -d connl-pe -v f1-alpha-match-10 # f1-macro
