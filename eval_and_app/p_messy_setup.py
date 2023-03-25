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

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.net = models.resnet50(pretrained=True)
    def forward(self, x):
        output = self.net.conv1(x)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        return output

def get_img_feature(imgPath):
    img = Image.open(imgPath)
    img = img.convert("RGB")
    img = transform(img)
    with torch.no_grad():
        x = img.unsqueeze(0).to(device)
        y = resnet(x)
        y = y.squeeze(2).squeeze(-1).squeeze(0)  # torch.Size([2048])
    return y

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

resnet = Resnet().to(device)

transform = transforms.Compose([
            transforms.ToTensor()]
        )
cnt = 0

questions = []
rel_cnt = 0

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
        if p == "child":
            continue
        if p == "spouse":
            continue
        if p not in label.keys():
            label.update({p: label_cnt})
            label_cnt += 1
print(label_cnt)

json.dump(label, open('data/{}/targets.json'.format(args.file), 'w'), indent=4)

for subset in ["train", "dev", "test"]:
    triples = []
    rel_counter = {}
    with open('{}.source'.format(subset)) as f:
        for line in f:
            line = line.strip()
            s, p, o = line.replace('_', ' ').split('\t')
            if p not in label.keys():
                continue
            triples.append((s, p, o))
            if p not in rel_counter.keys():
                rel_counter.update({p: 0})
            rel_counter[p] += 1
    print(rel_counter)

    img_path = []
    with open('{}.{}'.format(subset, args.type)) as f:
        for line in f:
            line = line.strip()
            if line.split('/')[-3] not in label.keys():
                continue
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
        sentence = s + " and " + o
        
        if subset == "train":
            num = int(rel_counter["team"]/rel_counter[p])
        else:
            num = 1
        for x in range(num):
            data.append([sentence, s, p, o, label[p], img])
    final_data = data
    print(len(data))
    
    with open('data/{}/{}.pkl'.format(args.file, subset), 'wb') as f:
        pickle.dump(final_data, f)
    
# with open('data/{}/{}/targets.json'.format(args.model, args.file)) as f:
#     f = json.load(f)
#     target_data = torch.zeros(len(f.keys()), 768)
    
with open('data/{}/vocab.pkl'.format(args.file[:9]), 'wb') as f:
    pickle.dump(torch.zeros(label_cnt), f)