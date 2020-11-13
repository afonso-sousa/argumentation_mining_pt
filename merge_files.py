import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get stats from free text file.",
                                     epilog="example: python3 merge_files.py *paths/to/corpora")
    parser.add_argument("corpora_paths", nargs="+")
    args = parser.parse_args()

    with open('data/merged_data.txt', 'w') as outfile:
        for fname in args.corpora_paths:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
