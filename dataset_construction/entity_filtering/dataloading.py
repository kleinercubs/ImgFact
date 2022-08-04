import os

class DataChunk():
    def __init__(self, path, chunkid, isbaddata) -> None:
        self.filepath = path
        self.chunkid = chunkid
        self.imagedata = dict()
        self.textdata = dict()
        self.isbaddata = isbaddata
    
    def load(self):
        if not self.isbaddata:
            record = open(self.filepath + "/logs/" + self.chunkid + "/record.txt", "r", encoding = "utf-8")
            records = record.readlines()
            for line in records:
                line = line.split("\t")
                if len(line) < 2:
                    continue
                if line[1] == "":
                    continue
                if int(line[1])  < 10:
                    continue
                try:
                    if not os.path.exists(self.filepath + "/crawleddata/" + self.chunkid + "/" + line[0]):
                        continue
                except:
                    continue
                self.imagedata[line[0]] = list()
                for i in range(int(line[1])):
                    path = self.filepath + "/crawleddata/" + self.chunkid + "/" + line[0]
                    filename = line[0] + "+" + str(i+1) + ".jpg"
                    self.imagedata[line[0]].append(os.path.join(path, filename))
                    
            textfile = os.path.join("entabstract", self.chunkid + ".txt")
            textfile = open(textfile, "r", encoding = "utf-8")
            for line in textfile.readlines():
                line = line[:-1].split("\t")
                if len(line) < 2:
                    continue
                entity = line[0]
                abstract = line[1]
                if not self.imagedata.__contains__(entity):
                    continue
                self.textdata[entity] = abstract
        else:
            textfile = os.path.join("entabstract", self.chunkid + ".txt")
            textfile = open(textfile, "r", encoding = "utf-8")
            for line in textfile.readlines():
                line = line[:-1].split("\t")
                if len(line) < 2:
                    continue
                entity = line[0]
                abstract = line[1]
                if os.path.exists(self.filepath + "/crawleddata/" + entity):
                    filenum = len(os.listdir(self.filepath + "/crawleddata/" + entity))
                    if filenum  < 10:
                        continue
                    self.imagedata[entity] = list()
                    for i in range(filenum):
                        path = self.filepath + "/crawleddata/" + entity
                        filename = entity + "+" + str(i+1) + ".jpg"
                        self.imagedata[entity].append(os.path.join(path, filename))
                    self.textdata[entity] = abstract             
                                
        
        return self.textdata, self.imagedata

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
