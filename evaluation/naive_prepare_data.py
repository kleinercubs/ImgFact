import torch
from transformers import AutoTokenizer, AdamW
from transformers import ViltProcessor, BertModel
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from PIL import Image
import time
import random
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import argparse
import ranger
from ranger import Ranger

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='vilt_app_s')
parser.add_argument('--device', type=str, default='0')

args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')   

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

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, s, p, o, label, path = self.data[index]
        image = torch.tensor(list(image_dict[path]))
        return [image, sentence, f'{s}/{p}/{o}']


def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    return batch

def calculate(pure_bert, resnet, loader, infer=False):
    tk = tqdm(loader, total=len(loader), position=0, leave=True)
    with torch.no_grad():
        for _, data in enumerate(tk):
            image, sentence, path = data
            image = torch.stack(list(image)).squeeze(1)
            des = tokenizer(list(sentence), max_length=40, padding='max_length', truncation=True, return_tensors='pt').to(device)
            input_ids = des['input_ids']  # [bs, sentence_len]
            text_emb = pure_bert(input_ids).pooler_output
            for image, text, ins in zip(image, text_emb, path):
                if ins not in vilt_dict.keys():
                    vilt_dict[ins] = []
                vilt_dict[ins].append([image, text.cpu()])

    
if __name__ == '__main__':
    with open('image_naive.pkl'.format(args.dataset), 'rb') as f:
        image_dict = pickle.load(f)
        
    with open('data/{}/train.pkl'.format(args.dataset), 'rb') as f:
        train_data = pickle.load(f)

    with open('data/{}/dev.pkl'.format(args.dataset), 'rb') as f:
        dev_data = pickle.load(f)

    with open('data/{}/test.pkl'.format(args.dataset), 'rb') as f:
        test_data = pickle.load(f)
        
    train_set = CustomDataset(train_data)
    val_set = CustomDataset(dev_data)
    test_set = CustomDataset(test_data)

    params = {
        'batch_size': 128,
        'shuffle': True,
        'num_workers': 16
    }


    train_loader = DataLoader(train_set, **params, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, **params, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, **params, collate_fn=collate_fn)

    
    resnet = Resnet().to(device)
    resnet.eval()
    pure_bert = BertModel.from_pretrained('bert-base-uncased').to(device)
    pure_bert.eval()
    for name, parameter in resnet.named_parameters():
        parameter.requires_grad = False
    for name, parameter in pure_bert.named_parameters():
        parameter.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    vilt_dict = dict()
    calculate(pure_bert, resnet, train_loader)
    calculate(pure_bert, resnet, val_loader)
    calculate(pure_bert, resnet, test_loader)
    
    for key in vilt_dict.keys():
        x, y = [], []
        for a, b in vilt_dict[key]:
            x.append(a)
            y.append(b)
        
        vilt_dict[key] = [torch.mean(torch.stack(x), dim=0), torch.mean(torch.stack(y), dim=0)]

    with open(f'data/{args.dataset}/image_naive.pkl', 'wb') as f:
        pickle.dump(vilt_dict, f)