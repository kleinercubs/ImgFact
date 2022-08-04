import os
import clip
import copy
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
img_to_tensor = transforms.ToTensor()

PROMPT_TOKENS = 10

def get_device(net):
    sd = net.state_dict()
    for v in sd.values():
        return v.device

class pTuningembedding(nn.Module):
    def __init__(self, CLIP):
        super(pTuningembedding, self).__init__()
        self.clipembedding = copy.deepcopy(CLIP.token_embedding)
        self.ptembedding = nn.Embedding(50010,512)
        self.clipembedding.requires_grad_(False)
        self.ptembedding.requires_grad_(True)
        self.device = get_device(CLIP)
    def forward(self, tokens):
        tokenNum = PROMPT_TOKENS
        mask = np.array([0] + [1 for i in range (tokenNum)] + [0 for i in range(76-tokenNum)])
        mask_org = 1-mask
        newtokens=torch.unsqueeze(tokens,0)
        newtokens = torch.transpose(newtokens, 2, 1)
        mask = torch.unsqueeze(torch.Tensor(mask).to(self.device),1)
        mask_org = torch.unsqueeze(torch.Tensor(mask_org).to(self.device),1)
        token = torch.transpose(newtokens*mask.long(), 2, 1)
        token_org = torch.transpose(newtokens*mask_org.long(), 2, 1)
        emb = self.ptembedding(torch.squeeze(token, 0))
        emb_org = self.clipembedding(torch.squeeze(token_org, 0))
        finalembbeding = emb_org * mask_org + emb * mask
        return finalembbeding


device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda",1)

def getPtuningCLIP(device):
    CLIP, preprocess = clip.load("../model/clip/ViT-B-32.pt", device=device)
    CLIP = CLIP.float()
    CLIP.token_embedding = pTuningembedding(CLIP)
    for i, child in enumerate(CLIP.children()):
        if i != 2:
            for param in child.parameters():
                param.requires_grad = False

    linear = torch.nn.Linear(1,1) 
    new_Wight = torch.Tensor(np.ones([1,1])) 
    linear.weight = torch.nn.Parameter(new_Wight)
    linear.bias = torch.nn.Parameter(torch.Tensor(np.array([-0.5])))
    CLIP.add_module("valclassifier", nn.Sequential(linear, nn.Sigmoid()))
    clipmodel = copy.deepcopy(CLIP)
    return clipmodel, preprocess

clipmodel, preprocess = getPtuningCLIP(device)

from typing import Any, Union, List
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = clip._tokenizer.encoder["<|startoftext|>"]
    eot_token = clip._tokenizer.encoder["<|endoftext|>"]
    pTuning_tokens = [50000+i for i in range(PROMPT_TOKENS)]
    all_tokens = [[sot_token] + pTuning_tokens + clip._tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def del_bar(text):
    state = 0
    start = -1
    end = -1
    for i, w in enumerate(text):
        if state == 0 and w == "(":
            start = i
            state = 1
        else:
            if state == 1 and w == ")":
                end = i
                state = 0
                break
    if start != -1:
        text2 = text[:start]
        if end != -1:
            text2 += text[end+1:]
        return del_bar(text2)
    return text

class dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #text processing
        imagepath, label = self.data[index]
        text = imagepath.split("/")[2]
        text = text.split(" ")
        if len(text) != 2:
            raise ValueError("Wrong entity text:",text)
        text = ", ".join(text)
        text = " ".join(text.split("_"))
        text = del_bar(text)
        textoutput = tokenize(text, truncate = True).squeeze(0)

        #Image loading
        image = Image.open(imagepath)
        image = image.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        imageout = preprocess(image)

        return (textoutput, imageout, int(label), index)


imagedic = dict()
def loadimages(path, father):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            imagedic[" ".join(file.split())] = list()
            loadimages(file_path, file)
        else:
            image = Image.open(file_path)
            image = image.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
            imagedic[" ".join(father.split())].append(image)

def test(model, data, bestf1, threshold = 0.5):
    with torch.no_grad():
        prec = 0
        positive_cnt = 0
        negative_cnt = 0
        TP = 0
        TN = 0
        badcases = []
        for (text, image, orglabel, index) in data:
            text, image, orglabel = text.to(device), image.to(device), orglabel.to(device)
            logits_per_image = model.encode_image(image)
            logits_per_text = model.encode_text(text)
            logits =  torch.mm(logits_per_image.to(torch.float32), logits_per_text.to(torch.float32).T)
            pred = torch.diag(logits)/100
            pred = model.valclassifier(pred.unsqueeze(-1)).squeeze(-1)
            labels = orglabel
            predlabel = [1 if item.item() > threshold else 0 for item in pred]
            index = np.array(index)
            for i, item in enumerate(zip(list(predlabel), list(labels))):
                if item[1].item() == 0:
                    negative_cnt += 1
                else:
                    positive_cnt += 1
                if item[0] == item[1].item():
                    if item[0] == 0:
                        TN += 1
                    else:
                        TP += 1
                    prec += 1
                else:
                    badcases.append(index[i])
        if TP == 0:
            print("0 Prec")
            prec = 0
        else:
            prec = TP/(TP + negative_cnt - TN)
            print("Test Prec:" + str(prec))

        recall = TP/positive_cnt
        if prec + recall == 0:
            f1 = 0
        else:
            f1 = 2*prec*recall/(prec+recall)
        print("Test Recall:" + str(recall))
        print("Test F1: " + str(f1))
        update = False
        if f1 > bestf1:
            bestf1 = f1
            update = True
    return bestf1, badcases, update, prec, recall, f1

import random

class dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #Text processing
        imagepath, imagetext, label = self.data[index]
        text = imagetext.split("/")[2]
        text = text.split(" ")
        if len(text) != 2:
            raise ValueError("Wrong entity text:",text)
        text = ", ".join(text)
        text = " ".join(text.split("_"))
        text = del_bar(text)
        textoutput = tokenize(text, truncate = True).squeeze(0)

        #Image loading
        image = Image.open(imagepath)
        image = image.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
        imageout = preprocess(image)

        return (textoutput, imageout, int(label), index)

class PtuningInfDataset(torch.utils.data.Dataset):
    def __init__(self, data, preprocess, TARGET_IMG_SIZE=224):
        self.data = data
        self.preprocess = preprocess
        self.TARGET_IMG_SIZE = TARGET_IMG_SIZE

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #Text processing
        pic_path, s, o, _, _ = self.data[index]
        text = ", ".join([s, o])
        text = " ".join(text.split("_"))
        text = del_bar(text)
        textoutput = tokenize(text, truncate = True).squeeze(0)
        #Image loading
        image = Image.open(pic_path)
        image = image.resize((self.TARGET_IMG_SIZE, self.TARGET_IMG_SIZE))
        imageout = self.preprocess(image)

        return (imageout, textoutput, index)

def loaddatafile(path, textfile, data_aug = False):
    textfile = open(textfile, "r", encoding="utf-8-sig")
    text = []
    for line in textfile.readlines():
        line = line[:-1].split("\t")
        text.append([path + "/" + line[0], path + "/" + line[0], int(line[1])])
        if data_aug:
            file = line[0].split("/")
            entities = file[1].split(" ")
            entities = " ".join([entities[1], entities[0]])
            file[1] = entities
            file = "/".join(file)
            text.append([path + "/" + line[0], path + "/" + file, int(line[1])])

    random.shuffle(text)
    return text

def balance(list):
    poslist = [item for item in list if item[-1] == 1]
    neglist = [item for item in list if item[-1] == 0]
    random.shuffle(neglist)
    random.shuffle(poslist)
    poslist =  poslist * int(len(neglist)/len(poslist))
    resultlist = neglist + poslist
    random.shuffle(resultlist)
    return resultlist

if __name__ == "__main__":
    PRfile = open("prlog_dataaug.txt", "w", encoding="utf-8")
    resultlist = []

    bestdic = {
        "prec":0.0,
        "recall":0.0,
        "f1":0.0
    }

    TARGET_IMG_SIZE = 224
    path = "promptSampleData"
    textdata = loaddatafile(path, "samplelabel.txt")
    path = "sampledataset"
    filtereddatatext = loaddatafile(path, "newlabel.txt")
    filtereddata = dataset(filtereddatatext)
    textdata = balance(textdata)
    alldataset = dataset(textdata)
    print("Traindata:{}\nTestdata:{}".format(len(alldataset), len(filtereddata)))
    batch_size = 32
    alldataset = DataLoader(alldataset, batch_size=batch_size, shuffle=True, pin_memory = True, num_workers = 8)

    filtereddataset = DataLoader(filtereddata, batch_size=batch_size, shuffle=False, pin_memory = True, num_workers = 8)

    CLIP = copy.deepcopy(clipmodel)
    CLIP.to(device).train()
    epochs = 20

    lossfunc = nn.MSELoss()
    optimizer = torch.optim.Adam(CLIP.parameters(), lr=1e-4)
    bestf1 = 0
    for ep in range(epochs):
        CLIP.train()
        print("Epoch: ", str(ep+1))
        bar = tqdm(total = len(alldataset))
        batch_loss = []
        prec = 0
        positive_cnt = 0
        negative_cnt = 0
        TP = 0
        TN = 0
        for batch, (text, image, orglabel,_) in enumerate(alldataset):
            text, image, orglabel = text.to(device), image.to(device), orglabel.to(device)

            logits_per_image = CLIP.encode_image(image)
            logits_per_text = CLIP.encode_text(text)
            logits =  torch.mm(logits_per_image.to(torch.float32), logits_per_text.to(torch.float32).T)
            pred = torch.diag(logits)/100
            pred = CLIP.valclassifier(pred.unsqueeze(-1)).squeeze(-1)
            loss_text = lossfunc(pred, orglabel.to(torch.float32))
            optimizer.zero_grad()
            loss_text.backward()
            batch_loss.append(loss_text.item())
            bar.set_description("Loss:" + str(sum(batch_loss)/len(batch_loss)) + ", Prec:" + str(round(prec,3)))
            optimizer.step()
            bar.update(1)
        bar.close()

        threshold = 0.5
        bestf1, badcases, update, prec, recall, f1 = test(CLIP.eval(), filtereddataset, bestf1, threshold)
        if update == True:
            bestmodel = CLIP

    modelPath = f"checkpoints/ptuning_{int(100*bestf1)}_{PROMPT_TOKENS}.pth"
    torch.save(bestmodel.state_dict(), modelPath)
    badcasefile = open("badcase_p.txt", "w")
    for idx in badcases:
        badcasefile.write(filtereddatatext[idx][0]+"\t"+str(filtereddatatext[idx][2])+"\n")
    print("Best test F1:"+str(bestf1))