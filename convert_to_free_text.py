import argparse
from pathlib import Path

from utils import read_doc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts free text from preprocessed AM corpus.",
                                     epilog="example: python3 convert_to_free_text.py path/to/corpus")
    parser.add_argument("corpus_path", type=Path)

    args = parser.parse_args()

    _, paragraphs = read_doc(args.corpus_path)
    text = " ".join(paragraphs.keys())

    file_name = args.corpus_path.stem + "_free_text.txt"
    with open(file_name, "w") as text_file:
        text_file.write(text)
