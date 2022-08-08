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
parser.add_argument('--file', type=str, default='_predict_s')
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
with open('data_v0419.source') as f:
    for line in f:
        line = line.strip()
        s, p, o = line.replace('_', ' ').split('\t')
        if s not in label.keys():
            label.update({s: label_cnt})
            label_cnt += 1
        if o not in label.keys():
            label.update({o: label_cnt})
            label_cnt += 1
print(label_cnt)

head_entity, tail_entity = set(), set()
for subset in ["train", "dev", "test"]:
    triples = []
    with open('{}_v0419.source'.format(subset)) as f:
        for line in f:
            line = line.strip()
            s, p, o = line.replace('_', ' ').split('\t')
            triples.append((s, p, o))

    img_path = []
    with open('{}_v0419.prefix'.format(subset)) as f:
        for line in f:
            line = line.strip()
            line = '/'.join(line.split('/')[-3:])
            img_path.append(line)
    
    rel_cluster = json.load(open("rel_cluster.json", "r"))
    length = len(img_path)
    for idx in range(length-1):
        cur_rel = img_path[idx].split('/')[-3]
        if idx == length-2:
            target_idx = idx+1
            target_rel = img_path[target_idx].split('/')[-3]
            if rel_cluster[cur_rel] != rel_cluster[target_rel]:
                tmp = img_path[idx]
                img_path[idx] = img_path[target_idx]
                img_path[target_idx] = img_path[idx]
            break
        for x in range(100):
            target_idx = random.randint(idx + 1, length-1)
            target_rel = img_path[target_idx].split('/')[-3]
            if rel_cluster[cur_rel] != rel_cluster[target_rel]:
                tmp = img_path[idx]
                img_path[idx] = img_path[target_idx]
                img_path[target_idx] = img_path[idx]
                break
            
    data = []
    for triple, img in tqdm(zip(triples, img_path)):
        s, p, o = triple
        _, template = rel2desc[p]
        if subset == "train":
            head_entity.add(s)
            tail_entity.add(o)
        if "predict_s" in args.file:
            # if s not in head_entity:
            #     continue
            sentence = template.format('[MASK]', o)
            data.append([sentence, o, p, s, label[s], img])
            # data.append([sentence, o, p, s, label[s], get_img_feature(img).cpu()])
        else:
            # if o not in tail_entity:
            #     continue
            sentence = template.format(s, '[MASK]')
            data.append([sentence, s, p, o, label[o], img])
            # data.append([sentence, s, p, o, label[o], get_img_feature(img).cpu()])


    final_data = data
    print(len(data))
    
    with open('data/{}/{}.pkl'.format(args.file, subset), 'wb') as f:
        pickle.dump(final_data, f)
