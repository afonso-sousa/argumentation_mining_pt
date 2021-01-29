cd tagger
python test.py trained/multi_en_vec_tagger.hdf5 ../data/pt_pe/pt_test.dat -d connl-pe --emb-fn embeddings/wiki.multi.pt.vec --emb-dim 300 # -v f1-macro
