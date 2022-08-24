import shutil
import os
from torchvision import models, transforms
from transformers import AutoTokenizer, BertModel
from tqdm import tqdm
from PIL import Image
import pickle
import torch
import torch.nn as nn    
import argparse         

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='s')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bert = BertModel.from_pretrained('bert-base-uncased').to(device)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

res = []
imgdic = dict()
for subset in ['data']:
    with open("{}_v0419.source".format(subset), 'r', encoding = 'utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            

with open('image_naive.pkl', 'wb') as f:
    pickle.dump(imgdic, f)