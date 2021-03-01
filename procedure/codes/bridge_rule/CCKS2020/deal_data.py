import os

def read_triple(file_path):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((h,r,t))
    return triples

def write_triple(storefile, triple_list):
    en_triple = []
    for lines in triple_list:
        a = lines[0]
        c = lines[2]
        en_triple.append((a,c))

    with open(storefile, 'a+') as fin:
        for entities in en_triple:
            a = entities[0]
            c = entities[1]
            string = str(a)+'\t'+str(c)+'\n'
            fin.write(string)

if __name__=='__main__':
    readfile = 'valid.txt'
    stroefile = 'valid_1.txt'
    triples = read_triple(readfile)
    write_triple(stroefile ,triples)
