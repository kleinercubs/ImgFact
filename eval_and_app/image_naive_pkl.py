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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = Resnet().to(device)
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

if not os.path.exists('naive_entity_dic.pkl'):
    entity_dic = dict()
    entity_counter = dict()
    with open("mmkg.txt", "r", encoding = "utf-8") as f:
        for d in tqdm(f.readlines()):
            entity, fp = d.strip().split('\t')
            if entity not in entity_counter.keys():
                entity_counter[entity] = 0
                entity_dic[entity] = []
            entity_dic[entity].append(get_img_feature(fp).cpu().numpy())
    with open('naive_entity_dic.pkl', 'wb') as f:
        pickle.dump(entity_dic, f)
        
print("Start Loading...")
with open('naive_entity_dic.pkl', 'rb') as f:
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
                imgdic.update({os.path.join(rel, so, idx): [entity_dic[s][entity_idx], get_img_feature(fp).cpu().numpy(),]})
            else:
                imgdic.update({os.path.join(rel, so, idx): [entity_dic[s][entity_idx],]})

if args.type == "enhance":
    fn = 'image_naive_enhance.pkl'
else:
    fn = 'image_naive.pkl'
with open(fn, 'wb') as f:
    pickle.dump(imgdic, f)

