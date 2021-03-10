cd tagger
python main.py --train ../data/pt_pe/train_pt.dat --dev ../data/pt_pe/dev_pt.dat --test ../data/pt_pe/test_pt.dat --data-io connl-pe --evaluator f1-alpha-match-10 --opt adam --lr 0.001 --save-best yes --patience 30 --rnn-hidden-dim 200 --emb-fn embeddings/skip_s100.txt --emb-dim 100 --epoch-num 200 --min-epoch-num 100 --freeze-word-embeddings True --model BiRNNCRF

