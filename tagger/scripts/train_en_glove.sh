cd tagger
python main.py --train ../data/persuasive_essays/Paragraph_Level/train.dat.abs --dev ../data/persuasive_essays/Paragraph_Level/dev.dat.abs --test ../data/persuasive_essays/Paragraph_Level/test.dat.abs --data-io connl-pe --evaluator f1-alpha-match-10 --opt adam --lr 0.001 --save-best yes --patience 40 --rnn-hidden-dim 200 --epoch-num 300 --min-epoch-num 200
