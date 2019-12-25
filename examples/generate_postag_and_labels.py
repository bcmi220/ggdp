import sys

def read_conll(file_name):
    with open(file_name, "r") as fin:
        data = fin.readlines()
    conll_data = []
    sent_data = []
    for line in data:
        if len(line.strip()) == 0:
            if len(sent_data) > 0:
                conll_data.append(sent_data)
                sent_data = []
        else:
            sent_data.append(line.strip().split("\t"))
    if len(sent_data) > 0:
        conll_data.append(sent_data)
    return conll_data

if __name__ == '__main__':

    train_file = sys.argv[1]
    postag_file = sys.argv[4]
    label_file = sys.argv[5]

    all_data = read_conll(train_file)

    postag_set = set()
    label_set = set()
    for sent in all_data:
        for line in sent:
            postag_set.add(line[4])
            label_set.add(line[7])

    postag_list = ['_'] + list(postag_set)
    label_list = ['_'] + list(label_set)

    with open(postag_file, "w") as fout:
        for item in postag_list:
            fout.write(item+"\n")

    with open(label_file, "w") as fout:
        for item in label_list:
            fout.write(item+"\n")

    