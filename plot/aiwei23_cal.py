import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer
plt.rcParams['font.size'] = 18  # 设置全局字体大小为14
from collections import defaultdict

watermark_types = ["john23","xuandong23b","aiwei23","rohith23","xiaoniu23","aiwei23b","scott22"]
replace_dict = {
    "john23": "TGRL",
    "xuandong23b": "UG",
    "aiwei23": "UPV",
    "rohith23": "RDF",
    "xiaoniu23": "UB",
    "lean23": "CTWL",
    "scott22": "GO",
    "aiwei23b": "SIR",
}


colors = [
    'darkorange',
    'deepskyblue',
    'limegreen',
    'violet',
    'goldenrod',
    'lightpink',

    'teal'
]


# attacks = ["ContractionAttack", "copypaste-3-10", "ExpansionAttack",  "MisspellingAttack", "synonym-0.4", "copypaste-1-10", "dipper_l20_o0", "LowercaseAttack", "swap", "TypoAttack"]
attacks = ["swap","translation","synonym-0.4", "copypaste-1-10","copypaste-3-10","copypaste-1-25","copypaste-3-25", "ContractionAttack", "ExpansionAttack",  "MisspellingAttack",   "dipper_l20_o0", "dipper_l40_o0",  "dipper_l60_o0",  "dipper_l40_o20", "dipper_l60_o20",     "dipper_l60_o40",    "LowercaseAttack", "TypoAttack"]
# attacks = ["ContractionAttack",  "ExpansionAttack",  "MisspellingAttack", "synonym-0.4",  "LowercaseAttack", "swap", "TypoAttack"]
# 常用颜色
# colors = ['b', 'g',  'c', 'm', 'y', 'k']
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

tpr_dict = defaultdict(dict)
dataset = 'c4'
model = 'opt'
# 创建一个空字典来存储roc_auc值
roc_auc_dict = {}

def read_file(file):
    data_list = []
    with open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'w_wm_output_attacked' in data:
                try:
                    length = data["w_wm_output_attacked_length"]
                except:
                    length = len(tokenizer(data["w_wm_output_attacked"])['input_ids'])
            elif 'w_wm_output' in data:
                try:
                    length = data["w_wm_output_length"]
                except:
                    length = len(tokenizer(data["w_wm_output"])['input_ids'])

            if length>150:
                data_list.append(json.loads(line))
    if len(data_list) ==0: 
        print(file)
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'w_wm_output_attacked' in data:
                    try:
                        length = data["w_wm_output_attacked_length"]
                    except:
                        length = len(tokenizer(data["w_wm_output_attacked"])['input_ids'])
                elif 'w_wm_output' in data:
                    try:
                        length = data["w_wm_output_length"]
                    except:
                        length = len(tokenizer(data["w_wm_output"])['input_ids'])

                if length>80:
                    data_list.append(json.loads(line))      
    data_list = data_list[:500]
    return data_list
watermark_type = 'aiwei23'
dataset = 'c4'
for j, attack in enumerate(attacks):
    # print(f"Processing {watermark_type} with {attack}...")
    try:
        data_list = read_file(f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/{dataset}/opt/{attack}/gen_table_w_metrics.jsonl')
    except:
        print(f"Missing {watermark_type} with {attack}...")
        continue
    
    if "baseline_completion_prediction" in data_list[0]:
        baseline_completion_prediction = [data["baseline_completion_prediction"] for data in data_list]
        w_wm_output_attacked_prediction = [data["w_wm_output_attacked_prediction"] for data in data_list]
    elif "baseline_completion_watermarked" in data_list[0]:
        baseline_completion_prediction = [data["baseline_completion_watermarked"] for data in data_list]
        w_wm_output_attacked_prediction = [data["w_wm_output_attacked_watermarked"] for data in data_list]
    else:
        print(f"load failed {watermark_type} with {attack}...")
        continue

    TP = w_wm_output_attacked_prediction.count(True)
    FN = w_wm_output_attacked_prediction.count(False)

    TPR = TP / (TP + FN)
    print(f'{watermark_type},{attack},{TPR}')