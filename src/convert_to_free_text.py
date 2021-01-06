import argparse
from pathlib import Path

from utils import read_doc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts free text from preprocessed AM corpus.",
                                     epilog="example: python3 convert_to_free_text.py free_text_corpus.txt path/to/corpus")
    parser.add_argument("output_path")
    parser.add_argument("corpus_path", type=Path)

    args = parser.parse_args()

    _, paragraphs = read_doc(args.corpus_path)
    text = " ".join(paragraphs.keys())

    # file_name = args.corpus_path.stem + "_free_text.txt"
    with open(args.output_path, "w") as text_file:
        text_file.write(text)

    print(f"File successfully converted to free text. Free-text file saved at \'{args.output_path}\'.")
