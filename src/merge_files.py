import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge files.",
                                     epilog="example: python merge_files.py merged_data.txt *paths/to/corpora")
    parser.add_argument("output_path")
    parser.add_argument("corpora_paths", nargs="+")
    args = parser.parse_args()

    with open(args.output_path, 'w') as outfile:
        for fname in args.corpora_paths:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    
    print(f"Files successfully merged. Merged file saved at \'{args.output_path}\'.")
