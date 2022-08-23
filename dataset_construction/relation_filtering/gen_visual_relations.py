from io import FileIO


allrel = open("RelationsSorted.txt", "r", encoding="utf-8")
visrel = open("VisualRelationsSorted.txt", "r", encoding="utf-8")

def readfile(file:FileIO):
    resultdic = dict()
    for line in file.readlines():
        line = line[:-1].split(" ")
        if len(line) < 2:
            continue
        resultdic[line[1]] = int(line[0])
    return resultdic

alldic = readfile(allrel)
visdic = readfile(visrel)


resultlist = list()
for key in visdic:
    resultlist.append((visdic[key]/alldic[key], key))

resultlist.sort(reverse=True)


resultfile = open("visrel.txt", "w", encoding = "utf-8")
for item in resultlist:
    if alldic[item[1]] < 20:
        continue
    resultfile.write(str(round(item[0],4)) + "\t" + str(alldic[item[1]]) + "\t" +  str(item[1]) + "\n")