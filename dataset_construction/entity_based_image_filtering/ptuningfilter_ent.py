from tqdm import tqdm
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from dataloading  import DataChunk
from ptuning_ent import getPtuningCLIP, PtuningInfDataset

PROMPT_TOKENS = 8

# Load the model
device = torch.device("cuda:0")
model, preprocess = getPtuningCLIP(device)
model.load_state_dict(torch.load("ptuningent/ptuning_87_44.pth"))
model.to(device)

def encode_text(CLIP, text, device):
    x = CLIP.token_embedding(text).type(CLIP.dtype)  # [batch_size, n_ctx, d_model]
    x = x + CLIP.positional_embedding.type(CLIP.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = CLIP.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = CLIP.ln_final(x).type(CLIP.dtype)
    additional_pos = torch.Tensor(np.array([PROMPT_TOKENS+1]*len(text))).long().to(device)
    argmax_res = text[:,PROMPT_TOKENS+1:].argmax(dim=-1) + additional_pos
    x = x[torch.arange(x.shape[0]), argmax_res] @ CLIP.text_projection

    return x


if __name__ == "__main__":
    THRESHOLD = 0.87

    DatasetPath = "ptuning_entfilterres"
    chunklist = ["Triplelist001", "Triplelist002", "Triplelist003", "Triplelist004", "Triplelist005",
                 "Triplelist011", "Triplelist012", "Triplelist013"]
    for chunk in chunklist:
        print("Filtering data:", chunk)
        imgdataset = DataChunk("/data/scrape_data", chunk).load()

        print("Image count:",str(len(imgdataset)))
        resultdata = [0 for i in range(len(imgdataset))]
        traindata = PtuningInfDataset(imgdataset, preprocess)
        batch_size = 64
        traindataset = DataLoader(traindata, batch_size=batch_size, shuffle=False, pin_memory = True, prefetch_factor=6, num_workers = 16)
        with torch.no_grad():
            bar = tqdm(total = len(traindataset))
            for (image, text1, text2, idxs) in traindataset:
                image, text1, text2 = image.to(device), text1.to(device), text2.to(device)
                endlabels = []
                orgentlabels = []
                logits_per_image = model.encode_image(image)
                for text in [text1, text2]:
                    logits_per_text = encode_text(model, text, device)
                    logits =  torch.mm(logits_per_image.to(torch.float32), logits_per_text.to(torch.float32).T)
                    # if torch.equal(logits_per_text[0], logits_per_text[1]):
                    #     print("YES")
                    pred = torch.diag(logits)/100
                    pred = model.valclassifier(pred.unsqueeze(-1)).squeeze(-1)
                    predlabel = [1 if item.item() >= THRESHOLD else 0 for item in pred]
                    orgentlabel = [item.item() for item in pred]
                    endlabels.append(predlabel)
                    orgentlabels.append(orgentlabel)
                # exit()
                predlabels = [endlabels[0][i]*endlabels[1][i] for i in range(len(endlabels[0]))]

                for i in range(len(image)):
                    if predlabels[i] == 0:
                        continue
                    resultdata[idxs[i]] = ((float(orgentlabels[0][i])+float(orgentlabels[1][i]))/2, float(orgentlabels[0][i]), float(orgentlabels[1][i]))
                bar.update(1)
            bar.close()

        resdic = dict()

        for i,data in enumerate(imgdataset):
            pic_path, s, o, father, file = data
            s = " ".join(s.split("_"))
            o = " ".join(o.split("_"))
            tupst = (str(s),str(father),str(o))
            if tupst not in resdic.keys():
                resdic[tupst] = []
            if resultdata[i] == 0:
                continue
            resdic[tupst].append((resultdata[i], file, pic_path))

        #rank the images and select topK images for each triplet
        keylist = [w for w in resdic.keys()]
        for key in keylist:
            resdic[key].sort(reverse = True)

        #Generate Dataset and LogFile
        if not os.path.exists('./' + DatasetPath):
            os.makedirs("./" + DatasetPath)
        logfile = open(DatasetPath + "/" + chunk + ".txt", "w", encoding = "utf-8")
        imgcnt = 0
        for key in resdic.keys():
            entity1 = "_".join(key[0].split(" "))
            entity2 = "_".join(key[2].split(" "))
            for index, tups in enumerate(resdic[key]):
                imgcnt += 1
                logfile.write(key[1] + "\t" + entity1 + "\t" + entity2 + "\t" + tups[1] + "\t" + str(round(tups[0][1], 3)) + "\t" + str(round(tups[0][2], 3)) + "\n")
        logfile.close()
        tqdm.write("Got {} Tuples, {} images".format(str(len(resdic.keys())), str(imgcnt)))