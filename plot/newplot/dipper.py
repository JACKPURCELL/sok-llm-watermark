import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer
plt.rcParams['font.size'] = 18  # 设置全局字体大小为14
from collections import defaultdict

watermark_types = ["john23","xuandong23b","rohith23","scott22"]
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
attacks = ["dipper_40_rep5"]
# attacks = ["ContractionAttack",  "ExpansionAttack",  "MisspellingAttack", "synonym-0.4",  "LowercaseAttack", "swap", "TypoAttack"]
# 常用颜色
# colors = ['b', 'g',  'c', 'm', 'y', 'k']
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

            if length>100:
                data_list.append(json.loads(line))
    if len(data_list) < 100: 
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

                if length>40:
                    data_list.append(json.loads(line))      
    data_list = data_list[:500]
    return data_list
fig, axs = plt.subplots(2, 4, figsize=(40, 20))
# 将 axs 转换为一维数组，以便我们可以迭代它
axs = axs.flatten()
bar_width = 0.35
opacity = 0.8
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
# tokenizer = AutoTokenizer.from_pretrained("meta-opt/opt-2-7b-chat-hf")
for i, watermark_type in enumerate(watermark_types):

    for j, attack in enumerate(attacks):
        # print(f"Processing {watermark_type} with {attack}...")
        try:
            data_list = read_file(f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/{dataset}/opt/{attack}/gen_table_w_metrics.jsonl')
        except:
            print(f"Missing {watermark_type} with {attack}...")
            continue
        
        count = 0
        for i in data_list:
            if i['w_wm_output_attacked_z_score'] > thresholds[watermark_type]:
                count += 1
        print(f"{watermark_type} with {attack} count: {count} list:{len(data_list)} rate: {count/len(data_list)}")