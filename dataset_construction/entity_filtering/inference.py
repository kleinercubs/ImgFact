import torch
from torchvision import models
import numpy as np
import os
from torch import nn
from PIL import Image
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import csv
import random
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import logging
import torchvision.transforms as transforms
import random
import time
import copy
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloading import DataChunk, DataCollection
import argparse
parser = argparse.ArgumentParser()

img_to_tensor = transforms.ToTensor()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, textdata, imagedata):
        self.textdata = textdata
        self.imagedata = imagedata
        self.entitylist = list(self.textdata.keys())

    def __len__(self):
        return len(self.entitylist)

    def __getitem__(self, index):
        entity = self.entitylist[index]
        
        #Text Processing
        text = self.textdata[entity]
        output = tokenizer.encode(text, padding=True, truncation=True, max_length = 64)
        entityid = tokenizer.encode(entity, padding=True, max_length = 64)
        pad = [0 for i in range(64)]
        output = output + pad
        entityid = entityid + pad
        textoutput = torch.tensor(output[:64])
        entityid = torch.tensor(entityid[:64])

        #Image loading
        images = self.imagedata[entity]
        imagedata = []
        for image in images:
            try:
                image = Image.open(image)
            except:
                print("Error image:", image)
                continue
            image = image.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
            if np.array(image).shape != (224,224,3):
                image = np.expand_dims(np.array(image), 2)
                image = np.concatenate((image, image, image), axis=2)
                image = Image.fromarray(image)
            
            tensor = img_to_tensor(image)
            imagedata.append(tensor)
            
        if len(imagedata) < 20:
            for i in range(20-len(imagedata)):
                imagedata.append(random.choice(imagedata))
        imageout = torch.stack(imagedata, dim=0)
        return (entityid, index, textoutput, imageout)


vgg16 = models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("../model/vgg16/vgg16.pth"))
vgg16 = vgg16.features


logging.set_verbosity_error()
model_name = '../model/bert-base-uncased'
start = time.time()
tokenizer = BertTokenizer.from_pretrained(model_name)
bertmodel = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
end = time.time()
print("Loading model cost: " + str(end - start))

class VCC(nn.Module):
    def __init__(self):
        super(VCC, self).__init__()
        self.vgglayer = copy.deepcopy(vgg16)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.bertlayer = copy.deepcopy(bertmodel.bert)
        self.vggfc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
        )
        self.multiimgfc = nn.Sequential(
            nn.Linear(256*20, 256),
        )
        self.bertfc = nn.Sequential(
            nn.Linear(768, 256),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256*2, 2),
        )
    def imageforward(self, imageinputs):
        output = []
        imageinputs = torch.transpose(imageinputs, 0, 1)
        for image in imageinputs:
            vggout = self.vgglayer(image)
            poolout = self.avgpool(vggout)
            poolout = torch.flatten(poolout, 1)
            imageout = self.vggfc(poolout)
            output.append(imageout)
        out = torch.cat(output, 1)
        return out           
    def forward(self, textinput, imageinputs):
        vggout = self.imageforward(imageinputs)
        imageout = self.multiimgfc(vggout)
        bertout = self.bertlayer(textinput).pooler_output
        textout = self.bertfc(bertout)
        output = self.classifier(torch.cat([imageout, textout], 1))
        return output


def loaddatafile(textfile):
    textfile = open(textfile, "r", encoding="utf-8-sig")
    reader = csv.reader(textfile)
    text = ["\t".join(line) for line in reader]
    random.shuffle(text)
    return text


if __name__ == "__main__":
    TARGET_IMG_SIZE = 224
    device = torch.device("cuda", 1)  

    vccmodel = VCC()
    vccmodel.load_state_dict(torch.load("../checkpoints/smallvccbest_0_11.pth"))
    vccmodel = vccmodel.to(device)
    vccmodel.eval()
    
    finishedLists = set()
    for file in os.listdir("vccresult"):
        filename = file[:-4]
        finishedLists.add(filename)

    filelist = ["111", "113", "121", "123", "53"]
    for file in filelist:
        workpath = os.path.join("../crawldata", file, "scrape")
        Col = DataCollection(workpath)
        datalist = Col.getChunks()
        for data in datalist:
            isbaddata = False
            if "../" in data:
                data = data[3:]
                isbaddata = True
            if data in finishedLists:
                continue
            
            chunk = DataChunk(workpath, data, isbaddata)
            textdata, imagedata = chunk.load()
            print("Predicting data:", workpath, " ",data)
            PredictRes = [0]*len(textdata)
            infdata = Dataset(textdata, imagedata)
            batch_size = 64
            dataset = DataLoader(infdata, batch_size=batch_size, shuffle=True, pin_memory = True, prefetch_factor=6, num_workers = 16)
            with torch.no_grad():
                bar = tqdm(total = len(dataset))
                for batch, (entity, idx, text, images) in enumerate(dataset):
                    text, images = text.to(device), images.to(device)
                    
                    predict = vccmodel(text, images)
                    predlabel = predict.softmax(dim=1).argmax(dim=1)
                    pred = np.array(predlabel.cpu())
                    ents = np.array(entity)
                    for id, item in enumerate(ents):
                        PredictRes[idx[id]] = pred[id]
                    bar.update(1)
                bar.close()
            predictdata = open("vccresult/"+data+".txt", "w", encoding = "utf-8")
            for i, line in enumerate(textdata):
                predictdata.write(line + "\t" + str(PredictRes[i]) + "\n")