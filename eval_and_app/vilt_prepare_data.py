import torch
from transformers import ViltProcessor, ViltForImagesAndTextClassification
from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import argparse
import random

random.seed(19260817)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='vilt_app_s')
parser.add_argument('--device', type=str, default='0')

args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.device)
                      if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-nlvr2")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, s, p, o, label, path = self.data[index]
        img = image_dict[path]
        if "random" in args.dataset:
            img = torch.rand((224, 224, 3))
        inputs = self.processor(img, sentence, max_length=40,
                                padding='max_length', truncation=True, return_tensors='pt')
        ids = inputs.input_ids
        att_mask = inputs.attention_mask
        pixel_values = inputs.pixel_values
        pixel_mask = inputs.pixel_mask
        # return [ids, att_mask, pixel_values, pixel_mask, f'{s}/{p}/{o}', inputs]
        return [ids, att_mask, pixel_values, pixel_mask, f'{s}/{p}/{o}']


def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*batch))
    return batch

def calculate(model, loader, infer=False):
    tk = tqdm(loader, total=len(loader), position=0, leave=True)
    with torch.no_grad():
        for _, data in enumerate(tk):
            ids, att_mask, pixel_values, pixel_mask, path = data
            ids = torch.stack(list(ids)).squeeze(1).to(device)
            att_mask = torch.stack(list(att_mask)).squeeze(1).to(device)
            pixel_values = torch.stack(
                list(pixel_values)).to(device)
            pixel_mask = torch.stack(list(pixel_mask)).to(device)
            if pixel_values is not None and pixel_values.ndim == 4:
                # add dummy num_images dimension
                pixel_values = pixel_values.unsqueeze(1)
            pooler_outputs = []
            num_images = pixel_values.shape[1] if pixel_values is not None else None
            for i in range(num_images):
                outputs = model(
                    ids,
                    pixel_values=pixel_values[:, i, :, :,
                                              :] if pixel_values is not None else None,
                    image_token_type_idx=i+1,
                )
                pooler_output = outputs[1]
                pooler_outputs.append(pooler_output)

            pooled_output = torch.cat(pooler_outputs, dim=-1)
            vecs = [pooled_output]
            for outs, ins in zip(vecs, path):
                if ins not in vilt_dict.keys():
                    vilt_dict[ins] = []
                vilt_dict[ins].append(outs.cpu())


if __name__ == '__main__':    
    if 'enhance' in args.dataset:
        image_pkl_file = 'image_vilt_enhance.pkl'
    else:
        image_pkl_file = 'image_vilt.pkl'
    with open(image_pkl_file, 'rb') as f:
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
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 16
    }

    train_loader = DataLoader(train_set, **params, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, **params, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, **params, collate_fn=collate_fn)

    model = ViltForImagesAndTextClassification.from_pretrained(
        "dandelin/vilt-b32-finetuned-nlvr2").vilt.to(device)

    vilt_dict = dict()
    calculate(model, train_loader)
    calculate(model, val_loader)
    calculate(model, test_loader)

    for key in vilt_dict.keys():
        vilt_dict[key] = torch.mean(torch.stack(vilt_dict[key]), dim=0)

    with open(f'data/{args.dataset}/image_vilt.pkl', 'wb') as f:
        pickle.dump(vilt_dict, f)
