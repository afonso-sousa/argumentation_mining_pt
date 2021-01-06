python main.py --train data/AM/persuasive_essays/Paragraph_Level/train.dat.abs --dev data/AM/persuasive_essays/Paragraph_Level/dev.dat.abs --test data/AM/persuasive_essays/Paragraph_Level/test.dat.abs --data-io connl-pe --evaluator f1-alpha-match-10 --opt adam --lr 0.001 --save-best yes --patience 20 --rnn-hidden-dim 200

