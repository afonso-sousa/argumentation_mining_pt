cd tagger
python run_tagger.py trained/skip_s100_200_2_tagger.hdf5 ../data/pt_pe/test_pt.dat -d connl-pe -v f1-alpha-match-10 # f1-macro
