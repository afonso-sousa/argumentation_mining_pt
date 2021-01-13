cd tagger
python main.py --train ../data/persuasive_essays/Paragraph_Level/train.dat.abs --dev ../data/persuasive_essays/Paragraph_Level/dev.dat.abs --test ../data/persuasive_essays/Paragraph_Level/test.dat.abs --data-io connl-pe --evaluator f1-alpha-match-10 --opt adam --lr 0.001 --save-best yes --patience 20 --rnn-hidden-dim 200 --emb-fn embeddings/wiki.multi.en.vec --emb-dim 300

