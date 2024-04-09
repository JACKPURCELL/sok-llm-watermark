
import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer
plt.rcParams['font.size'] = 18  # 设置全局字体大小为14
from collections import defaultdict
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

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

data_list_a = read_file("/home/ljc/sok-llm-watermark/runs/token_200/john23/c4/opt/gen_table_w_metrics.jsonl")
data_list_b = read_file("/home/ljc/sok-llm-watermark/runs/token_200/xuandong23b/c4/opt/gen_table_w_metrics.jsonl")


z_scores_a = [d['w_wm_output_z_score'] for d in data_list_a]
z_scores_b = [d['w_wm_output_z_score'] for d in data_list_b]

# z_scores_a = [d['w_wm_output_z_score'] for d in data_list_a]
# z_scores_b = [d['w_wm_output_z_score'] for d in data_list_b]



print(f'Mean: {np.mean(z_scores_a)}, Standard Deviation: {np.std(z_scores_a)}')
print(f'Mean: {np.mean(z_scores_b)}, Standard Deviation: {np.std(z_scores_b)}')