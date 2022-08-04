import os

class DataChunk():
    def __init__(self, path, chunkid) -> None:
        self.filepath = path
        self.chunkid = chunkid

    def load(self, result_type="list"):
        if result_type not in ["list", "dict"]:
            raise ValueError(f"Wrong result_type:{result_type}")
        reldict = dict()
        dataset = list()

        record = open(self.filepath + "/logs/" + self.chunkid + "/record.txt", "r", encoding = "utf-8")
        records = record.readlines()
        for line in records:
            line = line[:-1].split("\t")
            if len(line) < 4:
                continue
            if line[-1] == "":
                continue
            entity1 = "_".join(line[0].split(" "))
            entity2 = "_".join(line[1].split(" "))
            entitypair = " ".join([("_".join(line[0].split(" "))), ("_".join(line[1].split(" ")))])
            relation = line[2]
            imgpath = self.filepath + "/crawleddata/" + self.chunkid + "/" + relation + "/" + entitypair
            if not os.path.exists(imgpath):
                continue

            if result_type == "dict":
                if relation not in reldict:
                    reldict[relation] = dict()
                if entitypair not in reldict[relation]:
                    reldict[relation][entitypair] = list()

            for i in range(int(line[-1])):
                filename = str(i+1) + ".jpg"
                if result_type == "list":
                    dataset.append((os.path.join(imgpath,filename), entity1, entity2, relation, filename))
                if result_type == "dict":
                    reldict[relation][entitypair].append((os.path.join(imgpath,filename), entity1, entity2, relation, filename))

        if result_type == "dict":
            return reldict
        if result_type == "list":
            return dataset

class DataCollection():
    def __init__(self, path) -> None:
        self.folderpath = path
        self.data = dict()
        self.filelist = list()
        with open(os.path.join(self.folderpath, "finishEnts.txt"), "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.split("\t")
                if len(line) < 2:
                    continue
                filename = line[0][:-4]
                if not os.path.exists(os.path.join(self.folderpath, "crawleddata", filename)):
                    with open("lostchunk.txt", "a", encoding = "utf-8") as lf:
                        lf.write(filename + "\n")
                    filename = os.path.join("..", filename)
                self.filelist.append(filename)

    def getChunks(self):
        return self.filelist