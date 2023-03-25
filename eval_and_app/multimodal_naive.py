import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW, BertModel
import json
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image
import time
import random
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import pickle
import argparse
from ranger import Ranger
from sklearn.metrics import precision_score, f1_score, recall_score

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='bert-base-uncased')
parser.add_argument('--img_token_num', type=int, default=1)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--valid_batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optimizer', type=str, default="adamw")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_in_len', type=int, default=52)
parser.add_argument('--accum_iter', type=int, default=1)
parser.add_argument('--img_source', type=str, default='local')
parser.add_argument('--modality', type=str, default='naive')
parser.add_argument('--dataset', type=str, default='vilt_predict_s')
parser.add_argument('--device', type=str, default='0')

args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.device)
                      if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(args.seed)

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

class MyModel(nn.Module):
    def __init__(self, resnet, bert, tokenizer, class_num):
        super(MyModel, self).__init__()
        if 'enhance' in args.dataset:
            self.converter = nn.Linear(4096, args.img_token_num * 768)  # TODO
        else:
            self.converter = nn.Linear(2048, args.img_token_num * 768)  # TODO
        if args.modality == "naive":
            self.classifier = nn.Linear(768*(args.img_token_num+1), class_num)
        else:
            self.classifier = nn.Linear(768, class_num)
        self.resnet = resnet
        self.bert = bert
        self.pure_bert = pure_bert
        self.tokenizer = tokenizer
        self.class_num = class_num
        self.tokenizer_embeddings = self.bert.get_input_embeddings()
        self.transform = transforms.Compose([
            transforms.ToTensor()]
        )
 
    def get_img_feature(self, imgPath):
        if args.img_source == "local":
            img = Image.open(imgPath)
            img = img.convert("RGB")
            img = self.transform(img)
            with torch.no_grad():
                x = img.unsqueeze(0).to(device)
                y = self.resnet(x)
                y = y.squeeze(2).squeeze(-1).squeeze(0)  # torch.Size([2048])
            return y
        elif args.img_source == "random":
            return torch.rand(2048).to(device)
    
    def forward(self, x):
        b_text, b_s, b_p, b_o, b_labels, b_img = x # b_o is truth label
        if args.modality == "naive":
            img_features = torch.stack(list(b_img), dim=0).to(device)
            text_emb = torch.stack(list(b_text), dim=0).to(device)
            img_output = self.converter(img_features)

            inputs_embeds = torch.cat([img_output, text_emb], dim=1)
        else:
            text_emb = torch.stack(list(b_text), dim=0).to(device)
            inputs_embeds = torch.cat([text_emb], dim=1)
        cls_output = self.classifier(inputs_embeds)
        y = tokenizer(list(b_o), max_length=args.max_in_len-args.img_token_num,
                      padding='max_length', truncation=True, return_tensors='pt').to(device)

        labels = b_labels
        b_labels = torch.tensor(b_labels).unsqueeze(-1).to(device)
        one_hot_label = torch.zeros(b_labels.shape[0], class_num).to(
            device).scatter_(1, b_labels, 1)

        return cls_output, torch.tensor(labels), one_hot_label.to(device), b_s, b_p, b_o


class CustomDataset(Dataset):
    def __init__(self, data, train=False):
        self.data = set(['||||||'.join(d[:-2]+[str(d[-2])]) for d in data])
        self.data = [x.split('||||||') for x in self.data]
        if "predict_p" in args.dataset:
            rel_counter = dict()
            data = self.data
            self.data = []
            for d in data:
                if d[-1] not in rel_counter.keys():
                    rel_counter[d[-1]] = 0
                rel_counter[d[-1]] += 1
            mx_rel_counter = max(rel_counter.values())
            for d in data:
                if train:
                    num = int(mx_rel_counter/rel_counter[d[-1]])
                else:
                    num = 1
                for x in range(num):
                    self.data.append(d)
        if "random" in args.dataset:
            self.noise = torch.rand((len(data),  2048))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, s, p, o, label = self.data[index]
        label, path = int(label), f'{s}/{p}/{o}'
        if "random" in args.dataset:
            sentence = image_dict[path][1]
            image = self.noise[index]
        else:
            sentence = torch.tensor(list(image_dict[path][1]))
            image = torch.tensor(list(image_dict[path][0]))
        return [sentence, s, p, o, label, image]


def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    return batch


class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(epoch, model, loader, optimizer):
    print('epoch:', epoch)
    losses = AverageMeter()
    tk = tqdm(loader, total=len(loader), position=0, leave=True)
    loss_fn = nn.CrossEntropyLoss()
    
    for _, data in enumerate(tk):
        logits, labels, onehot_label, b_s, b_p, b_o = model(data)
        loss = loss_fn(logits, onehot_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item() * args.accum_iter, len(data))
        tk.set_postfix(loss=losses.avg)


def validate(epoch, model, loader):
    tk = tqdm(loader, total=len(loader), position=0, leave=True)
    cnt, acc1, acc5, mr, mrr, tot = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for _, data in enumerate(tk):
            logits, label, onehot_label, b_s, b_p, b_o = model(data)
            pred = torch.argsort(logits, descending=True)
            for pred, label in zip(pred.cpu(), label.cpu()):
                if label == pred[0]:
                    acc1 += 1
                if label in pred[:10]:
                    acc5 += 1
                mat = torch.tensor([label]*class_num)
                rank = torch.nonzero(mat == pred)[0][0]+1
                mrr += 1/rank
                mr += rank
                tot += 1
            tk.set_description("LP {} hit@1:{:.6f}   hit@10:{:.6f}   mrr:{:.6f}   mr:{:.6f}".format(
                args.dataset,
                acc1/tot,
                acc5/tot,
                mrr/tot,
                mr/tot,
            ))
    cnt = acc5
    return cnt


def inference(epoch, model, loader):
    # model.eval()
    cnt, acc1, acc5, mr, mrr, tot = 0, 0, 0, 0, 0, 0
    y_pred, y_true = [], []
    tk = tqdm(loader, total=len(loader), position=0, leave=True)
    f = open("output_dir/{}_{}_{}_{}_result.txt".format(args.dataset,
             args.modality, args.optimizer, args.lr), "w")
    with torch.no_grad():
        for _, data in enumerate(tk):
            logits, label, onehot_label, b_s, b_p, b_o = model(data)
            pred = torch.argsort(logits, descending=True)
            for pred, label, _s, _p, _o in zip(pred.cpu(), label.cpu(), b_s, b_p, b_o):
                if label == pred[0]:
                    acc1 += 1
                if label in pred[:10]:
                    acc5 += 1   
                mat = torch.tensor([label]*class_num)
                rank = torch.nonzero(mat==pred)[0][0]+1
                mrr += 1/rank
                mr += rank
                tot += 1
                y_true.append(int(label))
                y_pred.append(int(pred[0]))
                f.writelines('{}\t{}\t{}\t{}\n'.format(_s,_p,_o,rank))
            tk.set_description("LP {} hit@1:{:.6f}   hit@5:{:.6f}   mrr:{:.6f}   mr:{:.6f}".format(
                args.dataset,
                acc1/tot,
                acc5/tot,
                mrr/tot,
                mr/tot,
            ))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    f.writelines("LP {} hit@1:{:.6f}   hit@5:{:.6f}   mrr:{:.6f}   mr:{:.6f}   f1:{:.6f}   rec:{:.6f}   prec:{:.6f}".format(
        args.dataset,
        acc1/tot,
        acc5/tot,
        mrr/tot,
        mr/tot,
        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        recall_score(y_true, y_pred, average="weighted", zero_division=0),
        precision_score(y_true, y_pred, average="weighted", zero_division=0),
    ))
    return acc1/tot, acc5/tot, mrr/tot, mr/tot

    
if __name__ == '__main__':    
    with open('data/{}/image_naive.pkl'.format(args.dataset), 'rb') as f:
        image_dict = pickle.load(f)
    
    with open('data/{}/train.pkl'.format(args.dataset), 'rb') as f:
        train_data = pickle.load(f)

    with open('data/{}/dev.pkl'.format(args.dataset), 'rb') as f:
        dev_data = pickle.load(f)

    with open('data/{}/test.pkl'.format(args.dataset), 'rb') as f:
        test_data = pickle.load(f)
        
    train_set = CustomDataset(train_data, train=True)
    val_set = CustomDataset(dev_data)
    test_set = CustomDataset(test_data)

    train_params = {
        'batch_size': args.train_batch_size,
        'shuffle': True,
        'num_workers': 16
    }

    val_params = {
        'batch_size': args.valid_batch_size,
        'shuffle': False,
        'num_workers': 16
    }

    train_loader = DataLoader(train_set, **train_params, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, **val_params, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, **val_params, collate_fn=collate_fn)

    resnet = Resnet().to(device)
    resnet.eval()
    for name, parameter in resnet.named_parameters():
        parameter.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case=True)
    bert = AutoModelForMaskedLM.from_pretrained(args.model)
    pure_bert = BertModel.from_pretrained(args.model).to(device)
    bert.eval()
    pure_bert.eval()
    for name, parameter in bert.named_parameters():
        parameter.requires_grad = False
    
    for name, parameter in pure_bert.named_parameters():
        parameter.requires_grad = False

    with open('data/{}/vocab.pkl'.format(args.dataset[:9]), 'rb') as f:
        target_data = pickle.load(f)
        print('data/{}/vocab.pkl'.format(args.dataset[:9]))
        print(target_data.shape)
        class_num = target_data.shape[0]

    model = MyModel(resnet, bert, tokenizer, class_num=class_num).to(device)

    if args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'ranger':
        optimizer = Ranger(model.parameters(), lr=args.lr)

    checkpoint_path = f"output_dir/{args.dataset}_{args.modality}_{args.optimizer}_{args.lr}_classifier.pt"
    converter_checkpoint_path = f"output_dir/{args.dataset}_{args.modality}_{args.optimizer}_{args.lr}_converter.pt"
    best_score = 0
    for epoch in range(args.epochs):
        train(epoch, model, train_loader, optimizer)
        if (epoch + 1) % 1 == 0:
            valid_score = validate(epoch, model, val_loader)
            validate(epoch, model, test_loader)
            if valid_score >= best_score:
                best_score = valid_score
                torch.save(model.classifier.state_dict(), checkpoint_path)
                torch.save(model.converter.state_dict(),
                           converter_checkpoint_path)
                print('best_score:', best_score)

    model.classifier.load_state_dict(
        torch.load(checkpoint_path)
    )
    model.converter.load_state_dict(
        torch.load(converter_checkpoint_path)
    )

    cnt = inference(tokenizer, model, test_loader)
    print(cnt)
    print(len(test_data))
