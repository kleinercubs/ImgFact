import shutil
import os
from torchvision import models, transforms
from transformers import ViltProcessor
from tqdm import tqdm
from PIL import Image
import pickle
import torch
import torch.nn as nn        

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
for subset in ['data']:
    with open("{}_v0419.prefix".format(subset), 'r', encoding = 'utf-8') as f:
        for line in tqdm(f.readlines()):
            idx = line[:-1].split('/')[-1]
            so = line[:-1].split('/')[-2]
            rel = line[:-1].split('/')[-3]
            path = '/'.join(line[:-1].split('/')[:-3])
            fp = os.path.join('images_144', rel, so, idx)
            if not os.path.exists(fp):
                fp = os.path.join('images_118', rel, so, idx)
            imgdic.update({os.path.join(rel, so, idx): 
                        Image.open(fp).resize((640, 480), Image.ANTIALIAS).convert("RGB")})

with open('image_vilt.pkl', 'wb') as f:
    pickle.dump(imgdic, f)