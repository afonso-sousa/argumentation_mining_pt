cd tagger
python main.py --train ../data/pt_pe/pt_train.dat --dev ../data/pt_pe/pt_dev.dat --test ../data/pt_pe/pt_test.dat --data-io connl-pe --evaluator f1-alpha-match-10 --opt adam --lr 0.001 --save-best yes --patience 20 --rnn-hidden-dim 200 --emb-fn embeddings/wiki.multi.pt.vec --emb-dim 300 --emb-skip-first yes --freeze-word-embeddings True --model BiRNNCRF

