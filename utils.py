def read_doc(file_path, index0=1):
    argument_tuple = []
    free_text = {}
    h = []
    for line in open(file_path):
        line = line.strip()
        if line == "":
            if h != []:
                argument_tuple.append(h)
                string = " ".join([x[0] for x in h])
                free_text[string] = h
            h = []
        else:
            x = line.split("\t")
            word, label = x[index0], x[-1]
            h.append((word, label))
    if h != []:
        argument_tuple.append(h)
        string = " ".join([x[0] for x in h])
        free_text[string] = h
    return argument_tuple, free_text
