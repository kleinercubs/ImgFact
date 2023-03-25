import shutil
import os
from torchvision import models, transforms
from transformers import ViltProcessor
from tqdm import tqdm
from PIL import Image
import pickle
import torch
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='')
args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
transform = transforms.Compose([
    transforms.ToTensor()]
)

rel2desc = {}
with open('rel2desc.txt') as f:
    for line in f:
        line = line.strip()
        rel, template, label = line.split('\t')
        rel2desc[rel] = label
print(rel2desc)
res = []
imgdic = dict()

if not os.path.exists('vilt_entity_dic.pkl'):
    entity_dic = dict()
    entity_counter = dict()
    with open("mmkg.txt", "r", encoding = "utf-8") as f:
        for d in tqdm(f.readlines()):
            entity, fp = d.strip().split('\t')
            if entity not in entity_counter.keys():
                entity_counter[entity] = 0
                entity_dic[entity] = []
            entity_dic[entity].append(Image.open(fp).resize((640, 480), Image.ANTIALIAS).convert("RGB"))
    with open('vilt_entity_dic.pkl', 'wb') as f:
        pickle.dump(entity_dic, f)
        
print("Start Loading...")
with open('vilt_entity_dic.pkl', 'rb') as f:
    entity_dic = pickle.load(f)
print("Loaded")
entity_counter = dict()
    
for subset in ['data']:
    with open("{}.prefix".format(subset), 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            idx = line[:-1].split('/')[-1]
            so = line[:-1].split('/')[-2]
            rel = line[:-1].split('/')[-3]
            path = '/'.join(line[:-1].split('/')[:-3])
            s = so.split(' ')[0]
            if s not in entity_counter.keys():
                entity_counter[s] = 0
            _s = s.strip().split('\t')
            _fp = f"all_entities/{_s}/{entity_counter[s]}.jpg"
            entity_idx = entity_counter[s]
            entity_counter[s] = (entity_counter[s]+1)%len(entity_dic[s])
            fp = os.path.join('images', rel, so, idx)            
            
            if "enhance" == args.type:
                imgdic.update({os.path.join(rel, so, idx): [entity_dic[s][entity_idx], Image.open(fp).resize((640, 480), Image.ANTIALIAS).convert("RGB"),]})
            else:
                imgdic.update({os.path.join(rel, so, idx): entity_dic[s][entity_idx]})
                
if args.type == "enhance":
    fn = 'image_vilt_enhance.pkl'
else:
    fn = 'image_vilt.pkl'
with open(fn, 'wb') as f:
    pickle.dump(imgdic, f)
