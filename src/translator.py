import argparse
from pathlib import Path

import nltk.data
import torch
from nltk.tokenize import word_tokenize
from transformers import MarianMTModel, MarianTokenizer
import sys
# nltk.download('punkt')


def translate(texts, model, tokenizer, language="pt", device=torch.device('cuda')):
    # Prepare the text data into appropriate format for the model
    def template(
        text): return f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]

    # Tokenize the texts
    encoded = tokenizer.prepare_seq2seq_batch(
        src_texts, return_tensors="pt").to(device)

    # Generate translation using model
    translated = model.generate(**encoded)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(
        translated, skip_special_tokens=True)

    return translated_texts


def free_text_to_sentences(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    return tokenizer.tokenize(text)


def chunks(lst, n):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def chunks_list(lst, n):
    l = []
    for i in range(0, len(lst), n):
        l.append(lst[i:i + n])
    return l


def align_chunks(chunk, translated_chunk, file_path=None):
    assert len(chunk) == len(translated_chunk), f'{chunk}\n{translated_chunk}'
    with open(file_path, 'a+') as f:
        for i in range(len(chunk)):
            f.write("{:s} ||| {:s}\n".format(chunk[i], translated_chunk[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate free text from English to Romance language.",
                                     epilog="example: python translator.py path/to/corpus --src_lang en --trg_lang pt")
    parser.add_argument("corpus_path", type=Path)
    parser.add_argument("--src_lang", default='en', type=str)
    parser.add_argument("--trg_lang", default='pt', type=str)
    parser.add_argument("--chunk_size", default=100, type=int)
    args = parser.parse_args()
    args.device = torch.device('cuda')

    print('Loading pretrained model and tokenizer')
    if args.src_lang == 'en':
        TARGET_MODEL_NAME = 'Helsinki-NLP/opus-mt-en-ROMANCE'
    else:
        TARGET_MODEL_NAME = 'Helsinki-NLP/opus-mt-ROMANCE-en'
    target_tokenizer = MarianTokenizer.from_pretrained(TARGET_MODEL_NAME)
    target_model = MarianMTModel.from_pretrained(TARGET_MODEL_NAME).to(
        args.device).half()  # fp16 should save memory

    print('Tokenizing free text into sentences')
    with open(args.corpus_path, "r") as text_file:
        file_contents = text_file.read()
        text_sentences = [free_text_to_sentences(paragraph) for paragraph in file_contents.split('\t')]

    text_sentences = [sentence for paragraph in text_sentences for sentence in paragraph]

    file_name = args.corpus_path.with_suffix("").as_posix() + "_translated.txt"

    chunks = chunks_list(text_sentences, args.chunk_size)
    print('Translating sentences')
    for i, chunk in enumerate(chunks):
        translated_chunk = translate(
            chunk, target_model, target_tokenizer, language=args.trg_lang, device=args.device)

        # split ponctuation
        translated_chunk = [' '.join(word_tokenize(sentence))
                            for sentence in translated_chunk]
        align_chunks(chunk, translated_chunk, file_path=file_name)
        print("Chunk done [{}/{}].".format(i + 1, len(list(chunks))))
