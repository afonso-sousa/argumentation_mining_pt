cd tagger
python run_tagger.py trained/multi_en_vec_tagger.hdf5 ../data/en_pe/test.dat -d connl-pe -v f1-alpha-match-10 # f1-macro
