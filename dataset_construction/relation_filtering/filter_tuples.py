import csv
from tqdm import tqdm
from multiprocessing import Pool
csv.field_size_limit(500 * 1024 * 1024)

N = 8

def proc(params):
    fileno, visualset = params
    readbar = tqdm()
    f = open("DBpediaClean\DBpediaClean_NO"+str(fileno)+".csv","r",encoding = 'utf-8')
    lst = list()

    for line in f.readlines():
        linestr = line
        line = line[:-1]
        line = line.split("\t")
        if len(line) < 3:
            continue
        readbar.update(1)

        if line[0] not in visualset or line[2] not in visualset:
            continue
        lst.append(linestr)

    f.close()
    return lst

if __name__ == '__main__':

    vccres = open("visualentity.txt", "r", encoding="utf-8")
    visualset = set()
    for line in vccres.readlines():
        line = line.split("\t")
        if len(line) < 2:
            continue
        line[0] = "_".join(line[0].split(" "))
        visualset.add(line[0])
    
    params = [(i+1, visualset) for i in range(N)]
    with Pool(processes=N) as pool:
        result = pool.map(proc, params)
    
    writefile = open("visualtriples.txt", "w", encoding="utf-8")
    for lst in result:
        writefile.writelines(lst)