import shutil
import os
from torchvision import models, transforms
from transformers import ViltProcessor
from tqdm import tqdm
from PIL import Image
import pickle
import torch
import torch.nn as nn        

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
            imgdic.update({os.path.join(rel, so, idx): get_img_feature(fp).cpu().numpy()})

with open('image_naive.pkl', 'wb') as f:
    pickle.dump(imgdic, f)