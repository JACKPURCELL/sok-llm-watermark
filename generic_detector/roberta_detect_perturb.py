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

def save_false_idx(data,  method):
    save_path = "/home/jkl6486/sok-llm-watermark/runs/token_200/" + method + "/c4/opt/adv_perturb" + "/escape/gen_table_attacked.jsonl"
    with open(save_path, "a") as f:
        json.dump(data, f)
        f.write("\n")

#load arguments

# gpus = "0"
methods = [ "john23", "rohith23", "xuandong23b", "scott22", "aiwei23","aiwei23b","xiaoniu23"]
# methods = [ "aiwei23","aiwei23b","xiaoniu23"]
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
    count = 0
    g0s0 = 0
    g0s1 = 0
    g1s0 = 0
    g1s1 = 0
    data_path = "/home/jkl6486/sok-llm-watermark/runs/token_200/" + method + "/c4/opt/adv_perturb" + "/gen_table_w_metrics.jsonl"
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            spec = 1 if data["w_wm_output_attacked_z_score"]>thresholds[method] else 0
            gene = 0 if data["w_wm_output_attacked_after_change"] else 1
            count+=1
            if spec == 0 and gene == 0:
                g0s0 += 1
            elif spec == 0 and gene == 1:
                g1s0 += 1
            elif spec == 1 and gene == 0:
                g0s1 += 1
            else:
                g1s1 += 1
    
    # print("Method: ", method)
    #g0s0,g1s0,g0s1,g1s1
    # print("g0s0, g1s0, g0s1, g1s1: ")
    print("\"{}\": [{}, {}, {}, {}],".format(method,round(g0s0/count, 2), round(g1s0/count, 2), round(g0s1/count, 2), round(g1s1/count, 2)))
    # print("g0s0, g0s1, g1s0, g1s1: ")
    # print( round(g0s0/count, 2), round(g0s1/count, 2), round(g1s0/count, 2), round(g1s1/count, 2))
    # print("Total: ", count)
    
 