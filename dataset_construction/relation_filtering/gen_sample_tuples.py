visualRel = open("VisualRelationsSorted.txt", "r", encoding="utf-8")
RelDic = dict()
for line in visualRel.readlines():
    line = line[:-1].split(" ")
    if len(line) < 2:
        continue
    num = line[0]
    ent = line[1]
    if int(num) < 50:
        continue
    RelDic[ent] = list()

resultfile = open("relSelTups.txt", "w", encoding="utf-8")
visualTup = open("visualtriples.txt", "r", encoding = "utf-8")
import random
tups = visualTup.readlines()
random.shuffle(tups)
for line in tups:
    linestr = line
    line = line[:-1].split("\t")
    if len(line) < 3:
        continue
    if not RelDic.__contains__(line[1]):
        continue
    if len(RelDic[line[1]]) >= 50:
        continue
    RelDic[line[1]].append(linestr)
    resultfile.write(linestr)


