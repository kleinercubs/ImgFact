from tqdm import tqdm
import torch
import os
from torch.utils.data import DataLoader
from dataloading  import DataChunk

from ptuning import getPtuningCLIP, PtuningInfDataset

# Load the model
device = torch.device("cuda:1")
model, preprocess = getPtuningCLIP(device)
model.load_state_dict(torch.load("ptuning/ptuning_75.pth"))
model.to(device)

if __name__ == "__main__":
    THRESHOLD = 0.4

    DatasetPath = "ptuningfilterres"
    chunklist = ["Triplelist014", "Triplelist015", "Triplelist021", "Triplelist022", "Triplelist023", 
                 "Triplelist024", "Triplelist025", "Triplelist028", "Triplelist029"] # Filter 1

    for chunk in chunklist:
        print("Filtering data:", chunk)
        imgdataset = DataChunk("/data/scrape_data", chunk).load()

        print("Image count:",str(len(imgdataset)))
        resultdata = [0 for i in range(len(imgdataset))]
        traindata = PtuningInfDataset(imgdataset, preprocess)
        batch_size = 64
        traindataset = DataLoader(traindata, batch_size=batch_size, shuffle=False, pin_memory = True, prefetch_factor=6, num_workers = 16)
        with torch.no_grad():
            bar = tqdm(total = len(traindataset))
            for (image, text, idxs) in traindataset:
                image, text = image.to(device), text.to(device)
                logits_per_image = model.encode_image(image)
                logits_per_text = model.encode_text(text)
                logits =  torch.mm(logits_per_image.to(torch.float32), logits_per_text.to(torch.float32).T)
                pred = torch.diag(logits)/100
                pred = model.valclassifier(pred.unsqueeze(-1)).squeeze(-1)
                predlabel = [1 if item.item() > THRESHOLD else 0 for item in pred]

                for i in range(len(image)):
                    if predlabel[i] == 0:
                        continue
                    resultdata[idxs[i]] = float(pred[i])
                bar.update(1)
            bar.close()
        
        resdic = dict()
        
        for i,data in enumerate(imgdataset):
            pic_path, s, o, father, file = data
            s = " ".join(s.split("_"))
            o = " ".join(o.split("_"))
            tupst = (str(s),str(father),str(o))
            if tupst not in resdic.keys():
                resdic[tupst] = []
            if resultdata[i] == 0:
                continue
            resdic[tupst].append((resultdata[i], file, pic_path))

        #rank the images and select topK images for each triplet
        keylist = [w for w in resdic.keys()]
        for key in keylist:
            resdic[key].sort(reverse = True)
        
        #Generate Dataset and LogFile
        if not os.path.exists('./' + DatasetPath):
            os.makedirs("./" + DatasetPath)
        logfile = open(DatasetPath + "/" + chunk + ".txt", "w", encoding = "utf-8")
        imgcnt = 0
        for key in resdic.keys():
            entity1 = "_".join(key[0].split(" "))
            entity2 = "_".join(key[2].split(" "))
            for index, tups in enumerate(resdic[key]):
                imgcnt += 1
                logfile.write(key[1] + "\t" + entity1 + "\t" + entity2 + "\t" + tups[1] + "\t" + str(round(tups[0], 2)) + "\n")
        logfile.close()
        tqdm.write("Got {} Tuples, {} images".format(str(len(resdic.keys())), str(imgcnt)))