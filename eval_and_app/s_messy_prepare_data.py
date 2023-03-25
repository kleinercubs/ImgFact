from transformers import AutoTokenizer, BertModel
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
parser.add_argument('--file', type=str, default='predict_s/messy')
parser.add_argument('--img_token_num', type=int, default=2)
parser.add_argument('--max_in_len', type=int, default=52)
parser.add_argument('--type', type=str, default='prefix')
args = parser.parse_args()

rel2desc = {}
with open('rel2desc.txt') as f:
    for line in f:
        line = line.strip()
        rel, template, label = line.split('\t')
        rel2desc[rel] = (template, label)
        
label = {}
label_cnt = 0
with open('data.source') as f:
    for line in f:
        line = line.strip()
        s, p, o = line.replace('_', ' ').split('\t')
        if s not in label.keys():
            label.update({s: label_cnt})
            label_cnt += 1
print(label_cnt)
json.dump(label, open('data/{}/targets.json'.format(args.file[:9]), 'w'), indent=4)

head_entity, tail_entity = set(), set()
for subset in ["train", "dev", "test"]:
    triples = []
    with open('{}.source'.format(subset)) as f:
        for line in f:
            line = line.strip()
            s, p, o = line.replace('_', ' ').split('\t')
            triples.append((s, p, o))

    img_path = []
    with open('{}.{}'.format(subset, args.type)) as f:
        for line in f:
            line = line.strip()
            line = '/'.join(line.split('/')[-3:])
            img_path.append(line)
            
    data = []
    for triple, img in tqdm(zip(triples, img_path)):
        s, p, o = triple
        _, template = rel2desc[p]
        if subset == "train":
            head_entity.add(s)
            tail_entity.add(o)
        if "predict_s" in args.file:
            if s not in head_entity:
                continue
            sentence = template.format('[MASK]', o)
            data.append([sentence, o, p, s, label[s], img])
            # data.append([sentence, o, p, s, label[s], get_img_feature(img).cpu()])
        else:
            if o not in tail_entity:
                continue
            sentence = template.format(s, '[MASK]')
            data.append([sentence, s, p, o, label[o], img])
            # data.append([sentence, s, p, o, label[o], get_img_feature(img).cpu()])


    final_data = data
    print(len(data))
    
    with open('data/{}/{}.pkl'.format(args.file, subset), 'wb') as f:
        pickle.dump(final_data, f)

if not os.path.exists('data/{}/vocab.pkl'.format(args.file[:9])):
    with open('data/{}/targets.json'.format(args.file[:9])) as f:
        f = json.load(f)
        target_data = torch.zeros(len(f.keys()), 768)
        print(len(f.keys()))
    with open('data/{}/vocab.pkl'.format(args.file[:9]), 'wb') as f:
        pickle.dump(target_data, f)