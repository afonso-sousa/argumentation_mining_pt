# %%
import argparse
from pathlib import Path

import nltk.data
from transformers import MarianMTModel, MarianTokenizer

# nltk.download('punkt')


def translate(texts, model, tokenizer, language="pt", device='cuda'):
    # Prepare the text data into appropriate format for the model
    def template(
        text): return f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]

    # Tokenize the texts
    encoded = tokenizer.prepare_seq2seq_batch(src_texts).to(device)

    # Generate translation using model
    translated = model.generate(**encoded)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(
        translated, skip_special_tokens=True)

    return translated_texts


def free_text_to_sentences(file_path):
    with open(file_path, "r") as text_file:
        file_contents = text_file.read()
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        return tokenizer.tokenize(file_contents)


def chunks(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def align_chunks(chunk, translated_chunk, file_path=None):
    assert len(chunk) == len(translated_chunk)
    with open(file_path, 'a+') as f:
        for i in range(len(chunk)):
            f.write("{:s} ||| {:s}\n".format(chunk[i], translated_chunk[i]))


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate free text from English to Romance language.",
                                     epilog="example: python3 translator.py path/to/corpus --src_lang en --trg_lang pt")
    parser.add_argument("corpus_path", type=Path)
    parser.add_argument("--src_lang", default='en', type=str)
    parser.add_argument("--trg_lang", default='pt', type=str)
    parser.add_argument("--chunk_size", default=100, type=int)
    args = parser.parse_args()

    args.device = 'cuda'

    print('Loading pretrained model and tokenizer')
    TARGET_MODEL_NAME = 'Helsinki-NLP/opus-mt-en-ROMANCE'
    target_tokenizer = MarianTokenizer.from_pretrained(TARGET_MODEL_NAME)
    target_model = MarianMTModel.from_pretrained(TARGET_MODEL_NAME).to(
        args.device).half()  # fp16 should save lots of memory

    print('Tokenizing free text into sentences')
    text_sentences = free_text_to_sentences(args.corpus_path)

    file_name = args.corpus_path.stem.split('_')[0] + "_translated.txt"

    chunks = chunks(text_sentences, args.chunk_size)
    print('Translating sentences')
    for i, chunk in enumerate(chunks):
        translated_chunk = translate(
            chunk, target_model, target_tokenizer, device=args.device)
        align_chunks(chunk, translated_chunk, file_path=file_name)
        print("Chunk done [{}/{}].".format(i, len(list(chunks))))


"""
print('Tokenizing free text into sentences')
text_sentences = free_text_to_sentences('data/merged_data.txt')
#nltk_text = nltk.data.load(text_sentences)
print(len(text_sentences))

# file_name = args.corpus_path.stem.split('_')[0] + "_translated.txt"

print('Translating sentences')
test = []
for chunk in chunks(text_sentences, 400):
    # translated_chunk = translate(
    #      chunk, target_model, target_tokenizer, device=args.device)
    # with open(file_name, 'w') as f:
    translated_chunk = ['asd', 'qwe', 'zcx', 'dfg', 'tyu', 'ghj', 'bnm', 'ert']
    align_chunks(chunk, translated_chunk)
    # for i, sentence in enumerate(translated_chunk):
    #     print("{:s} ||| {:s}\n".format(chunk[i], sentence))
    #     test.append("{:s} ||| {:s}\n".format(chunk[i], sentence))
    # f.write("{:s} ||| {:s}\n".format(chunk[i], sentence))
    print("Chunk done")

"""
