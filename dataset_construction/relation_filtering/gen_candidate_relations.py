import csv
from tqdm import tqdm
from multiprocessing import Pool
csv.field_size_limit(500 * 1024 * 1024)

N = 1

filename = 'DbpediaClean.csv'
name, filetype = filename.split(".")

def proc(fileno):
    readbar = tqdm()
    f = open("visualtriples.txt","r",encoding = 'utf-8')
    dic = dict()
    t = 1
    line = [1]
    for line in f.readlines():
        readbar.update(1)
        x = line.split("\t")
        line = x
        if len(line) < 3:
            continue
        if t == 1:
            t = 0
            continue
        if dic.__contains__(line[1]):
            dic[line[1]] = dic[line[1]] + 1
        else:
            dic[line[1]] = 1
    f.close()
    print("\nFile NO."+str(fileno)+" finished\n")
    return dic

if __name__ == '__main__':
    fileno = [i+1 for i in range(N)]
    with Pool(processes=N) as pool:
        result = pool.map(proc, fileno)
    newdict = dict()
    for dic in result:
        for w in dic.keys():
            if newdict.__contains__(w):
                newdict[w] = newdict[w] + dic[w]
            else:
                newdict[w] = dic[w]
    
    list = [(newdict[w], w) for w in newdict.keys()]
    print(len(list))
    # exit()
    list.sort(reverse=True)
    
    with open('VisualRelationsSorted.txt','w',encoding='utf-8') as f1:
        for w in list:
            if (w[0] >= 0):
                f1.write(str(w[0])+" "+str(w[1])+"\n")
            else:
                break
