import os

def read_file(readfile):
    with open(readfile, 'r') as fin:
        item_list = list()
        for line in fin:
            id, item = line.strip().split('\t')
            item_list.append(item)
    return item_list

def store_txt(item_list, storefile):
    with open(storefile, 'w') as fin:
        for item in item_list:
            string_in = str(item) + '\t' + str(item_list.index(item)) + '\n'
            fin.write(string_in)

if __name__ == '__main__':
    rd_file = 'entities.dict'
    se_file = 'entities.txt'
    r2_file = 'relations.dict'
    s2_file = 'relations.txt'
    entities = read_file(rd_file)
    store_txt(entities, se_file)
    relations = read_file(r2_file)
    store_txt(relations, s2_file)