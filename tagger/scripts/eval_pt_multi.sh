cd tagger
python run_tagger.py trained/multi_pt_vec_tagger.hdf5 ../data/pt_pe/test_pt.dat -d connl-pe -v f1-alpha-match-10 # f1-macro
