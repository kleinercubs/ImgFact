
import os

selectedResults = open("SelectedResults.txt", "w", encoding = "utf-8")
results = {"predict_s":{}, "predict_o":{}, "predict_p":{}}
for task in ["predict_s", "predict_o", "predict_p"]:
    path = os.path.join("output_dir", task)
    for file in os.listdir(path):
        if file.endswith(".txt"):
            # {args.dataset}_{args.modality}_{args.optimizer}_{args.lr}
            f = open(os.path.join(path, file),  "r", encoding = "utf-8")
            file = file[:-4].split('_')
            sub_task = file[0] # dataset
            model = file[1] # modality
            settings = '_'.join(file[2:]) #settings
            if settings not in results[task]:
                results[task][settings] = dict()
            if model not in results[task][settings]:
                results[task][settings][model] = dict()
            if sub_task not in results[task][settings][model]:
                line = f.readlines()[-1]
                results[task][settings][model][sub_task] = [
                    float(line.split('   ')[i].split(':')[1]) for i in range(7)
                ]

inputs = {
    "predict_s": ["messy", "p", "o", "spo"],
    "predict_p": ["messy", "s", "o", "spo"],
    "predict_o": ["messy", "s", "p", "spo"],
}

def better_metrics(task, x, y):
    cnt = 0
    if task == "predict_p":
        cnt += int(x[0] < y[0]) # hit1
        cnt += int(x[4] < y[4]) # f1
        cnt += int(x[5] < y[5]) # rec
        cnt += int(x[6] < y[6]) # prec
    else:
        cnt += int(x[0] < y[0]) # hit1
        cnt += int(x[1] < y[1]) # hit5
        cnt += int(x[2] < y[2]) # mrr
        cnt += int(x[3] > y[3]) # mr
    return cnt >= 2

# for task in ["predict_s", "predict_o", "predict_p"]:

for task in ["predict_s"]:
    for settings in results[task].keys():
        text_metric = []
        good_metric = 10
        for model in ["naive", "vilt"]:
        # for model in ["text", "naive", "vilt"]:
            if model == "text":
                text = results[task][settings]["text"]["spo"]
                continue
            # text vs multimodal
            # messy < text
            if not better_metrics(task, results[task][settings][model]["messy"], results[task][settings]["text"]["spo"]):
                good_metric -= 1
            # text < multimodal_entity
            for sub_task in inputs[task][1:-1]:
                if not better_metrics(task, results[task][settings]["text"]["spo"], results[task][settings][model][sub_task]):
                    good_metric -= 1
            # multimodal
            # entity < spo
            for sub_task in inputs[task][1:-1]:
                if not better_metrics(task, results[task][settings][model][sub_task], results[task][settings][model]["spo"]):
                    good_metric -= 1
        # if good_metric >= 8:
        if good_metric >= 0:
            selectedResults.writelines(f"Good_Metric:{good_metric}/10\n")
            for model in ["text", "naive", "vilt"]:
                if model == "text":
                    res = results[task][settings]["text"]["spo"]
                    selectedResults.writelines(f"{task} {settings} {model} spo hit@1:{res[0]} hit@5:{res[1]} mrr:{res[2]} mr:{res[3]} f1:{res[4]} rec:{res[5]} prec:{res[6]}\n")
                    continue
                for sub_task in inputs[task]:
                    res = results[task][settings][model][sub_task]
                    selectedResults.writelines(f"{task} {settings} {model} {sub_task} hit@1:{res[0]} hit@5:{res[1]} mrr:{res[2]} mr:{res[3]} f1:{res[4]} rec:{res[5]} prec:{res[6]}\n")
selectedResults.close()