cd tagger
python test.py trained/multi_en_vec_tagger.hdf5 ../data/pt_pe/test_pt.dat -d connl-pe --emb-fn embeddings/wiki.multi.pt.vec --emb-dim 300 -v f1-macro # f1-alpha-match-10 # f1-macro
