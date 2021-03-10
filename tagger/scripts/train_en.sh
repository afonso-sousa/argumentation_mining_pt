cd tagger
python main.py --train ../data/en_pe/train.dat --dev ../data/en_pe/dev.dat --test ../data/en_pe/test.dat --data-io connl-pe --evaluator f1-alpha-match-10 --opt adam --lr 0.001 --save-best yes --patience 30 --rnn-hidden-dim 200 --emb-fn embeddings/wiki.multi.en.vec --emb-dim 300 --emb-skip-first yes --epoch-num 200 --min-epoch-num 100 --freeze-word-embeddings True --model BiRNNCRF

