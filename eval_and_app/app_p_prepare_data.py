from transformers import AutoTokenizer
from torchvision import models, transforms
import json
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import argparse
from PIL import Image
import pickle
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab = tokenizer.vocab

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='vilt_app_p')
args = parser.parse_args()

rel2desc = {}
with open('rel2desc.txt') as f:
    for line in f:
        line = line.strip()
        rel, template, label = line.split('\t')
        rel2desc[rel] = (template, label)

label = {}
triples = []
triplet_set = set()
label_cnt = 0
with open('data_v0419.source') as f:
    for line in f:
        line = line.strip()
        s, p, o = line.replace('_', ' ').split('\t')
        triples.append((s, p, o))
        triplet_set.add((s,p,o))
        if s not in label.keys():
            label.update({s: label_cnt})
            label_cnt += 1
        if o not in label.keys():
            label.update({o: label_cnt})
            label_cnt += 1
print(label_cnt)

def get_diff_img(s,p,o,imgdic):
    imgs = imgdic[p]
    for img in imgs:
        if img[0] != s or img[1] != o:
            path = img[2]
            imgdic[p].remove(img)
            return path
    print(len(imgs), p)
    path = list(imgs)[0]
    imgdic[p].remove(path)
    return path[2]

train_o_set = set()
for subset in ["train", "dev", "test"]:
    triples = []
    idxs = []
    with open('{}_v0419.source'.format(subset)) as f:
        for idx, line in enumerate(f.readlines()):
            line = line.strip()
            s, p, o = line.replace('_', ' ').split('\t')
            if subset == "train":
                train_o_set.add(o)
            else:
                if o not in train_o_set:
                    idxs.append(idx)
                    continue
                    # print("{} not in train".format(o))
            triples.append((s, p, o))

    img_path = []
    with open('{}_v0419.prefix'.format(subset)) as f:
        for idx, line in enumerate(f.readlines()):
            if idx in idxs:
                continue
            line = line.strip()
            line = '/'.join(line.split('/')[-3:])
            img_path.append(line)
    data = []
    imgdic = dict()
    cnt = 0
    for idx, (triple, img) in enumerate(zip(triples, img_path)):
        s, p, o = triple
        _, template = rel2desc[p]
        if p not in imgdic:
            imgdic[p] = set()
        if "predict_s" in args.file:
            sentence = template.format('[MASK]', o)
            data.append((sentence, o, p, s, label[s], img))
        else:
            sentence = template.format(s, '[MASK]')
            data.append((sentence, s, p, o, label[o], img))
            if p == "occupation":
                cnt += 1
        imgdic[p].add((s,o,img, idx))

    print(cnt)
    cnt = 0
    for i in tqdm(range(1, len(data))):
        sentence, s, p, o, thislabel, img = data[i]
        if p == "occupation":
            cnt += 1
    print(cnt, len(imgdic["occupation"]))

    final_data = []
    spo_record = set()
    print(len(data))
    for i in tqdm(range(1, len(data))):
        sentence, s, p, o, thislabel, img = data[i]
        diff_img = get_diff_img(s,p,o, imgdic)
        if diff_img == None:
            print(imgdic[p])
            raise ValueError(f"no image for spo:{s} {p} {o}")
        final_data.append([sentence, s, p, o, thislabel, diff_img])

    with open('data/{}/{}.pkl'.format(args.file, subset), 'wb') as f:
        pickle.dump(final_data, f)
    print(len(final_data))
