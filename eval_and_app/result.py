# import os
# from select import select

# results = {"predict_s":{}, "predict_o":{}, "predict_p":{}}
# for task in ["predict_s", "predict_o", "predict_p"]:
#     path = os.path.join("output_dir", task)
#     for file in os.listdir(path):
#         if file.endswith(".txt"):
#             f = open(os.path.join(path, file),  "r", encoding = "utf-8")
#             file = file[:-4].split('_')
#             sub_task = file[0]
#             model = file[1]
#             settings = '_'.join(file[2:])
#             if sub_task not in results[task]:
#                 results[task][sub_task] = dict()
#             if settings not in results[task][sub_task]:
#                 results[task][sub_task][settings] = dict()
#             if model not in results[task][sub_task][settings]:
#                 line = f.readlines()[-1]
#                 print(line)
#                 results[task][sub_task][settings][model] = [
#                     float(line.split('   ')[i].split(':')[1]) for i in range(4)
#                 ]
                
# selectedResults = open("SelectedResults.txt", "w", encoding = "utf-8")

# for task in ["predict_s", "predict_o", "predict_p"]:
#     for sub_task in results[task].keys():
#         if sub_task == "spo":
#             continue
#         for settings in results[task][sub_task].keys():
#             flag = False
#             for model in ["vilt", "naive"]:
#                 cnt = 0
#                 print(task, sub_task, settings, model)
#                 cnt += results[task][sub_task][settings][model][0] >= results[task]['spo'][settings]['text'][0] 
#                 cnt += results[task][sub_task][settings][model][1] >= results[task]['spo'][settings]['text'][1]
#                 cnt += results[task][sub_task][settings][model][2] >= results[task]['spo'][settings]['text'][2]
#                 cnt += results[task][sub_task][settings][model][3] <= results[task]['spo'][settings]['text'][3]
#                 if cnt < 2:
#                     flag = True
#                     break
#             if not flag:
#                 for model in ['vilt', 'naive']:
#                     selectedResults.writelines("{}_{}_{}_{}".format(task, sub_task, settings, model)) 
#                     for i in range(4):
#                         selectedResults.writelines(" {}".format(results[task][sub_task][settings][model][i]))
#                     selectedResults.writelines("\n")
#                 selectedResults.writelines("{}_{}_{}_{}".format(task, sub_task, settings, 'text')) 
#                 for i in range(4):
#                     selectedResults.writelines(" {}".format(results[task]['spo'][settings]['text'][i]))
#                 selectedResults.writelines("\n\n")
#     selectedResults.writelines("\n{} DONE\n\n".format(task))
# selectedResults.close()

import os
from select import select

selectedResults = open("SelectedResults.txt", "w", encoding = "utf-8")
results = {"predict_s":{}, "predict_o":{}, "predict_p":{}}
for task in ["predict_s", "predict_o", "predict_p"]:
    path = os.path.join("output_dir", task)
    for file in os.listdir(path):
        if file.endswith(".txt"):
            f = open(os.path.join(path, file),  "r", encoding = "utf-8")
            file = file[:-4].split('_')
            sub_task = file[0]
            model = file[1]
            settings = '_'.join(file[2:])
            if sub_task not in results[task]:
                results[task][sub_task] = dict()
            if settings not in results[task][sub_task]:
                results[task][sub_task][settings] = dict()
            if model not in results[task][sub_task][settings]:
                line = f.readlines()[-1]
                selectedResults.writelines(f'{task} {sub_task} {settings} {model} {line}\n')
                results[task][sub_task][settings][model] = [
                    float(line.split('   ')[i].split(':')[1]) for i in range(4)
                ]
selectedResults.close()