import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import random
import torch
from torchvision import transforms, models
import os
import json
import numpy as np
import collections
from PIL import Image

from tqdm import tqdm
import torch
import argparse 
random.seed(998244353)
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="0")
parser.add_argument("--path", default="./")
args = parser.parse_args()
device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu") 
# device = torch.device("cpu") 
fw = None
f_selected = None
clip_score = None

ROOT_DIR = os.getcwd()
PATH = args.path

relation_num = 0
label = {}

avg_cluster = 0
deleted = 0
total_num = 0
cluster_score = 0


model = models.vgg19(pretrained=True).to(device)
model.classifier = model.classifier[:2]
for name, parameter in model.named_parameters():
    parameter.requires_grad = False
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
])

selectedRel = dict()
with open('/home/dell/lwc/cpgen/finalrels.txt', 'r', encoding = 'utf-8') as f:
    for line in f.readlines():
        line = line.strip().split('\t')
        selectedRel[line[0]] = int(line[1])==1

def listdir(file_path, flowers):
    with open(file_path, 'r', encoding = 'utf-8') as f:
        x = 0
        data = []
        for line in f.readlines():
            x = x + 1
            if x % 3 == 1:
                data.append(line[:-1])
                continue
            if x % 3 == 0:
                data = []
                continue
            data.append(line[:-1])
            s, p, o = data[0].split('\t')
            if not selectedRel[p]:
                continue
            idx = list(json.loads(data[1]).keys())
            if (s,p,o) not in idx:
                flowers.update({(s,p,o):[]})
            flowers[(s,p,o)] += idx

def extract_features(img, model):
    img = Image.open(img).convert("RGB").resize((224,224))
    img = transform(img).unsqueeze_(dim=0).to(device)
    ret = model(img)[0]
    return ret.cpu().numpy()


def dbscan(data,precomputed=False,eps=1,min_samples=1):
    if precomputed:
        model = DBSCAN(eps=eps,min_samples=min_samples,metric="precomputed")
    else:
        model = DBSCAN(eps=eps,min_samples=min_samples,metric="cosine")
    y_pred = model.fit_predict(data)
    all_y = list(y_pred)
    return [int(x) for x in all_y]

def calculate_diversity(info, cur_data, eps):
    global avg_cluster
    global total_num
    global deleted
    global cluster_score
    global fw
    global f_selected
    selected_cnt = 0
    labels = dbscan(cur_data,precomputed=False,eps=eps,min_samples=1)
    count = collections.Counter(labels)
    n_clusters = np.unique(labels).size
    avg_cluster += n_clusters
    deleted += labels.count(0)
    total_num = total_num + 1
    file_label = []
    f_selected.writelines(info[0][0] + '\t' + info[0][1] + '\t' + info[0][2] + '\n')
    
    visit = [0 for x in cur_data]
    for pos, val in enumerate(info):
        filename = '/data/scrape_data/crawleddata/{}/{}/{} {}/{}'.format(file[:-4],val[0],val[1],val[2], val[3])
        if visit[labels[pos]] == 0:
            file_label.append(val[3])
            f_selected.writelines("<img src=\"" + filename + "\" height=\"100\"/> ")
            selected_cnt += 1
            visit[labels[pos]] = 1
    f_selected.writelines('\n')
    tmp_score = None
    if (n_clusters == 1 or n_clusters == len(cur_data)):
        pass
    else:
        tmp_score = silhouette_score(cur_data, labels,metric="euclidean")
        cluster_score += tmp_score
    fw.writelines(info[0][0] + '\t' + info[0][1] + '\t' + info[0][2] + '\n')
    fw.writelines('\t'.join(file_label[0]))
    fw.writelines('\n{} {} {} {}\n'.format(n_clusters, len(visit) - n_clusters, tmp_score, len(cur_data)))
    fw.flush()
    return n_clusters, len(visit) - n_clusters, tmp_score, selected_cnt

fw = None
if args.device == "0":
    filelist = ["001", "003", "004", "005", "011", "012", "013", "014", "025"]
else:
    filelist = ["002", "015", "021", "022", "023", "024", "028", "029"]
eps = 0.25
all_selected, all_tot = 0, 0
for file in filelist: 
    file = "Triplelist{}.txt".format(file)
    file_path = os.path.join(PATH, file) 
    clip_score = {}
    with open(os.path.join('/home/dell/zmc/ClipFilter/ptuning_agg', file), 'r', encoding = 'utf-8') as f:
        for line in f.readlines():
            line = line[:-1].split('\t')
            p, s, o = line[:3]
            idx = line[3]
            score = (float(line[4]) + float(line[5]) + float(line[6])) / 3
            clip_score.update({(s, p, o, idx): score})
    fw = open('imgfact_v0419_0.3/{}.txt'.format(file[:-4]), 'w', encoding = 'utf-8')
    print(file[:-4] + '_selected.md')
    f_selected = open('imgfact_v0419_0.3/'+file[:-4] + '_selected.md', 'w', encoding = 'utf-8')
    flowers = dict()
    listdir(file_path, flowers)


    for _idx in idx:
        try:
            filename = '/data/scrape_data/crawleddata/{}/{}/{} {}/{}'.format(file[:-4],p,s,o,_idx)
            feat = extract_features(filename,model)
            cudf_list.append(((s, p, o, idx), feat, clip_score[(s, p, o, _idx)]))
        # if something fails, save the extracted features as a pickle file (optional)
        except :
            print('error:  ', flower)
            # err.writelines(str(flower) + "\n")
    filename = '/data/scrape_data/crawleddata/{}/{}/{} {}/{}'.format(file[:-4],p,s,o,idx)
    # # try to extract the features and update the dictionary
    cudf_list = sorted(cudf_list, key=lambda x:-x[2])
    cudf_name, cudf_list, _ = zip(*cudf_list)
    _n_clusters, _deleted, _tmp_score, _selected = calculate_diversity(cudf_name, cudf_list, eps)
    avg_num += _n_clusters
    avg_del += _deleted
    avg_selected += _selected
    if _tmp_score != None:
        avg_score += _tmp_score
    tot += 1
    bar.set_description('{}: avg_cluster_num: {} deleted:{} score:{}'.format(
        eps, avg_num/tot, avg_del/tot, avg_score/tot
    ))
    
    all_selected += avg_selected
    all_tot += tot
print(f'{all_selected}/{all_tot} = {all_selected/all_tot}')

