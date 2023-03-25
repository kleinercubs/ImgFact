import json
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import argparse
import pickle
import os

random.seed(998244353)

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='vilt_app_s')
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
with open('data.source') as f:
    for line in f:
        line = line.strip()
        s, p, o = line.replace('_', ' ').split('\t')
        if p == "child":
            continue
        if p == "spouse":
            continue
        triples.append((s, p, o))
        triplet_set.add((s,p,o))
        if p not in label.keys():
            label.update({p: label_cnt})
            label_cnt += 1
print(label_cnt)

json.dump(label, open('data/{}/targets.json'.format(args.file[:9]), 'w'), indent=4)

def get_diff_img(s,p,o,imgdic):
    imgs = imgdic[s]
    for img in imgs:
        # if rel_cluster[img[0]] != rel_cluster[p] or img[1] != o:
        if img[0] != p and img[1] != o:
            path = img[2]
            imgdic[s].remove(img)
            return True, path
    path = list(imgs)[0]
    imgdic[s].remove(path)
    return True, path[2]

rel_cluster = json.load(open("rel_cluster.json", "r"))
train_o_set = set()
for subset in ["train", "dev", "test"]:
    triples = []
    idxs = []
    none_diff = 0
    rel_counter = {}
    with open('{}.source'.format(subset)) as f:
        for idx, line in enumerate(f.readlines()):
            line = line.strip()
            s, p, o = line.replace('_', ' ').split('\t')
            if p not in label.keys():
                continue
            triples.append((s, p, o))
            if p not in rel_counter.keys():
                rel_counter.update({p: 0})
            rel_counter[p] += 1
    print('rel_counter:', rel_counter)

    img_path = []
    with open('{}.prefix'.format(subset)) as f:
        for idx, line in enumerate(f.readlines()):
            # if idx in idxs:
            #     continue
            line = line.strip()
            if line.split('/')[-3] not in label.keys():
                continue
            line = '/'.join(line.split('/')[-3:])
            img_path.append(line)
    data = []
    imgdic = dict()
    cnt = 0
    all_data = dict()
    for idx, (triple, img) in enumerate(zip(triples, img_path)):
        s,p,o = triple
        if (s,p,o) not in all_data.keys():
            all_data.update({(s,p,o):[]})
        all_data[(s,p,o)].append(img)
    for idx, cur_spo in enumerate(all_data.keys()):
        s, p, o = cur_spo
        _, template = rel2desc[p]
        if "/o" in args.file:
            if o not in imgdic:
                imgdic[o] = []
            sentence = s + " and " + o
            data.append((sentence, o, p, s, label[p], all_data[cur_spo]))
            imgdic[o].append((p,s,all_data[cur_spo]))
        else:
            if s not in imgdic:
                imgdic[s] = []
            sentence = s + " and " + o
            data.append((sentence, s, p, o, label[p], all_data[cur_spo]))
            imgdic[s].append((p,o,all_data[cur_spo]))

    final_data = []
    spo_record = set()
    new_rel_counter = dict()
    print('data:', len(data))
    new_data = []
    for i in tqdm(range(0, len(data))):
        sentence, s, p, o, thislabel, img = data[i]
        random.shuffle(imgdic[s])
        flag, diff_img = get_diff_img(s,p,o, imgdic)
        for img in diff_img:
            new_data.append([sentence, s, p, o, thislabel, img])
            if p not in new_rel_counter.keys():
                new_rel_counter.update({p: 0})
            new_rel_counter[p] += 1

    mx_rel_counter = max(new_rel_counter.values())
    for i in new_data:
        sentence, s, p, o, thislabel, diff_img = i
        if subset == "train":
            num = int(mx_rel_counter/new_rel_counter[p])
        else:
            num = 1
        for x in range(num):
            final_data.append([sentence, s, p, o, thislabel, diff_img])

    with open('data/{}/{}.pkl'.format(args.file, subset), 'wb') as f:
        pickle.dump(final_data, f)
    print('new_rel_counter:', new_rel_counter)
    print('balanced_data:', len(final_data))

# for x in final_data:
#     print(x)
with open('data/{}/targets.json'.format(args.file[:9])) as f:
    f = json.load(f)
    target_data = torch.zeros(len(f.keys()), 768)
    # for idx, x in tqdm(enumerate(list(f.keys()))):
    #     y = tokenizer(list(x), max_length=args.max_in_len-args.img_token_num, padding='max_length', truncation=True, return_tensors='pt').to(device)
    #     y = bert(y.input_ids).pooler_output[0]
    #     target_data[f[x]] = y
with open('data/{}/vocab.pkl'.format(args.file[:9]), 'wb') as f:
    pickle.dump(target_data, f)