import os
import json

def dataload(filename, storename):
    f = open(filename, encoding = 'UTF-8')
    setting = json.load(f)
    #print(setting)
    data = setting['relationships']
    with open(storename, 'a') as store:
        for lines in data:
            string = "\t".join(str(i) for i in lines)
            store.write(string+'\n')
    store.close()

def datareload(filename):
    f = open(filename, encoding='UTF-8')
    olddata = f.readlines()
    print('aa')
    f.close()



def datastrength(filename,strengthname):
    new_data = list()
    with open(filename, 'r', encoding='UTF-8') as fin:
        origin_data = fin.readlines()
        for origin in origin_data:
            origin_da = origin.strip('\n').split('\t')
            print(origin_da)
            if origin_da[1] == 'binding' or origin_da[1]== 'interaction':
                temp1 = list()
                temp2 = list()
                temp1.append(origin_da[0])
                temp1.append(origin_da[1])
                temp1.append(origin_da[2])
                temp2.append(origin_da[2])
                temp2.append(origin_da[1])
                temp2.append(origin_da[0])
                new_data.append(temp1)
                new_data.append(temp2)
            else:
                new_data.append(origin_da)
    return new_data


if __name__ == '__main__':
    filename = 'train.txt'
    #storename = 'test1.txt'
    #dataload(filename,storename)
    storename = 'str_train.txt'
    newlist = datastrength(filename,storename)
    #print(newlist)