import sys
"""
def read_doc(file_path):
    free_text = {}
    h = []
    for line in open(file_path):
        line = line.strip()
        if line == "":
            if h != []:
                string = " ".join([x[0] for x in h])
                free_text[string] = h
            h = []
        else:
            x = line.split("\t")
            word, label = x[1], x[-1]
            h.append((word, label))
    if h != []:
        string = " ".join([x[0] for x in h])
        free_text[string] = h
    return free_text
"""

def read_doc(file_path):
    free_text = []
    curr_parag_tuples = []
    num_parag = 0
    conll = open(file_path).readlines()
    for idx, line in enumerate(conll):
        if line.strip():
            idx_word_label = line.strip().split("\t")
            word, label = idx_word_label[1], idx_word_label[-1]
            curr_parag_tuples.append((word, label))
        else: # paragraph end
            if curr_parag_tuples != []:
                num_parag += 1
                string = " ".join([word_label[0] for word_label in curr_parag_tuples])
                free_text.append((string, curr_parag_tuples))
            curr_parag_tuples = []
    if curr_parag_tuples != []:
        num_parag += 1
        string = " ".join([word_label[0] for word_label in curr_parag_tuples])
        free_text.append((string, curr_parag_tuples))
    print(f'Total number of paragraphs: {num_parag}')
    return free_text