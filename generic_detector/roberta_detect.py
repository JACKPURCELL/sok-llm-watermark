"""
Finetune RoBERTa detector
"""
import argparse
import os
import torch
import numpy as np
import pandas as pd
import transformers
import json

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification, DataCollatorWithPadding, AutoModelForSequenceClassification

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import RobertaTokenizer
from transformers import TrainingArguments, Trainer

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from openai import OpenAI
import random

def save_false_idx(data, attack_epoch, method):
    save_path = "/home/jkl6486/sok-llm-watermark/runs/token_200/" + method + "/c4/opt/dipper_" + lex_len + "_rep" + str(attack_epoch) + "/escape/gen_table_attacked.jsonl"
    with open(save_path, "a") as f:
        json.dump(data, f)
        f.write("\n")

#load arguments

# gpus = "0"
methods = [ "john23", "rohith23", "xuandong23b", "scott22"]
lex_len = "40"
thresholds = {
    "john23": 2.358025378986253,
    "xuandong23b": 2.545584412271571,
    "aiwei23": 24.998546600341797,
    "rohith23": 1.8526251316070557,
    "xiaoniu23": 0.00,
    "lean23": 0.984638512134552,
    "scott22": 0.17697394677108003,
    "aiwei23b": 0.2496753585975497
}
# os.environ['CUDA_VISIBLE_DEVICES'] = gpus 
device = torch.device("cuda")

for method in methods:
    model_path = "/home/jkl6486/sok-llm-watermark/runs/token_200/" + method + "/c4/opt/dipper_l" + lex_len + "_o0/dipp_roberta_finetuned_chatgpt_new"
    #idx_path = "/home/jkl6486/sok-llm-watermark/runs/token_200/" + method + "/c4/opt/dipper_20_rep2/gen_table_filtered.jsonl"
    clf = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    ### Build the directories
    for i in range(2, 6):
        esc_path = "/home/jkl6486/sok-llm-watermark/runs/token_200/" + method + "/c4/opt/dipper_" + lex_len + "_rep" + str(i) + "/escape"
        os.makedirs(esc_path, exist_ok=True)
        for filename in os.listdir(esc_path):
            file_path = os.path.join(esc_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    ### Read the TP data from the first attack
    # idx_list = []
    # with open(idx_path, "r") as f:
    #     for line in f:
    #         data = json.loads(line)
    #         idx_list.append(data["idx"])

    ### Build the TP data list
    i = 0
    idx_list = []
    data_path = "/home/jkl6486/sok-llm-watermark/runs/token_200/" + method + "/c4/opt/dipper_l" + lex_len + "_o0/gen_table_attacked.jsonl"
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            x = tokenizer(data["w_wm_output_attacked"], return_tensors="pt", max_length=512, truncation=True).to(device)
            output = clf(**x)
            if output.logits.argmax(-1) == 1:
                idx_list.append(data["idx"])
                i += 1
          

    negative_count_list = [0]
    negative_idx_list = []
    result_list = []
    for attack_epoch in range(2,6):
        result_sub_list = []
        negative_idx_sub_list = []
        negative_count = 0
        data_path = "/home/jkl6486/sok-llm-watermark/runs/token_200/" + method + "/c4/opt/dipper_" + lex_len + "_rep" + str(attack_epoch) + "/gen_table_attacked.jsonl"
        with open(data_path, "r") as fd:
            for line in fd:
                data = json.loads(line)
                if data["idx"] in idx_list:
                    x = tokenizer(data["w_wm_output_attacked"], return_tensors="pt", max_length=512, truncation=True).to(device)
                    output = clf(**x)
                    if output.logits.argmax(-1) == 0:
                        negative_count += 1
                        negative_idx_sub_list.append(data["idx"])
                        save_false_idx(data, attack_epoch, method)
                        idx_list.remove(data["idx"])
                    result_sub_list.append(output.logits.argmax(-1).item())
        result_list.append(result_sub_list)
        negative_idx_list.append(negative_idx_sub_list)
        negative_count_list.append(negative_count_list[-1] + negative_count)
    print(method)
    print([_/i for _ in negative_count_list])
    print(i)
    print(negative_count_list)
    print()



                    


# method_names = ["john23", "xuandong23b"]

# thresholds = {
#     "john23": 2.358025378986253,
#     "xuandong23b": 2.545584412271571,
#     "aiwei23": 24.998546600341797,
#     "rohith23": 1.8526251316070557,
#     "xiaoniu23": 0.00,
#     "lean23": 0.984638512134552,
#     "scott22": 0.17697394677108003,
#     "aiwei23b": 0.2496753585975497
# }

# for method_name in method_names:
#     data_path = "/home/jkl6486/sok-llm-watermark/runs/token_200/" + method_name + "/c4/opt/dipper_l20_o0/gen_table_w_metrics.jsonl"
#     output_file_path = "/home/jkl6486/sok-llm-watermark/runs/token_200/" + method_name + "/c4/opt/dipper_20_rep2/gen_table_filtered.jsonl"

#     i = 0
#     with open(data_path, "r") as f, open(output_file_path, 'w') as outfile:
#         for line in f:
#             data = json.loads(line)
#             if data["w_wm_output_attacked_z_score"] > thresholds[method_name]:
#                 json.dump(data, outfile)
#                 outfile.write("\n")
#                 i += 1
#             if i == 100:
#                 break

        












