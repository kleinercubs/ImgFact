from transformers import BertModel
from torchvision import models, transforms
import json
from tqdm import tqdm
import random
import os
import torch
import torch.nn as nn
import argparse
from PIL import Image
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='_predict_s')
parser.add_argument('--model', type=str, default='vilt')
parser.add_argument('--img_token_num', type=int, default=2)
parser.add_argument('--max_in_len', type=int, default=52)
parser.add_argument('--type', type=str, default='prefix')
args = parser.parse_args()

cnt = 0

PATH = "/home/dell/lwc/cluster/imgfact"

questions = []
rel2id = {}
rel_cnt = 0

rel2desc = {}
with open('rel2desc.txt') as f:
    for line in f:
        line = line.strip()
        rel, template, label = line.split('\t')
        rel2desc[rel] = (template, label)

with open('data_v0419.source') as f:
    for line in f:
        line = line.strip()
        s, p, o = line.split('\t')
        if p == "child":
            continue
        if p == "spouse":
            continue
        if p not in rel2id.keys():
            rel2id.update({p: rel_cnt})
            rel_cnt += 1
print(rel_cnt)
json.dump(rel2id, open('rel2id.json', 'w'))

for subset in ['train', 'dev', 'test']:
    triples = []
    rel_counter = {}
    with open('{}_v0419.source'.format(subset)) as f:
        for line in f:
            line = line.strip()
            s, p, o = line.split('\t')
            s = s.replace('_', ' ')
            o = o.replace('_', ' ')
            p = p.replace('_', ' ')
            if p not in rel2id.keys():
                continue
            triples.append((s, p, o))
            if p not in rel_counter.keys():
                rel_counter.update({p: 0})
            rel_counter[p] += 1

    print([rel_counter])
    print(len(set(triples)))

    img_path = []
    with open('{}_v0419.prefix'.format(subset)) as f:
        for line in f:
            line = line.strip()
            if line.split('/')[-3] not in rel2id.keys():
                continue
            line = '/'.join(line.split('/')[-3:])
            img_path.append(line)
    
    data = []
    for triple, img in zip(triples, img_path):
        s, p, o = triple
        label, template = rel2desc[p]
        sentence = s + " and " + o
        if subset == "train":
            num = int(rel_counter["team"]/rel_counter[p])
        else:
            num = 1
        for x in range(num):
            data.append([sentence, s, p, o, rel2id[p], img])

    final_data = data
    print(len(data))
    
    with open('data/{}/{}.pkl'.format(args.file, subset), 'wb') as f:
        pickle.dump(final_data, f)
    
# with open('data/{}/{}/targets.json'.format(args.model, args.file)) as f:
#     f = json.load(f)
#     target_data = torch.zeros(len(f.keys()), 768)
    
with open('data/{}/vocab.pkl'.format(args.file[:9]), 'wb') as f:
    pickle.dump(torch.zeros(rel_cnt), f)