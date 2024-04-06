import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer
import os

import shutil
plt.rcParams['font.size'] = 18  # 设置全局字体大小为14
from collections import defaultdict
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
# watermark_types = ["aiwei23"]
watermark_types = ["john23","xuandong23b","aiwei23","rohith23","xiaoniu23","aiwei23b","scott22"]
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
dataset = 'c4'
# def read_file(file,watermark_type):
#     data_list = []
#     with open(file, 'r') as f:
#         for line in f:
#             data = json.loads(line)
#             if 'w_wm_output' in data:
#                 try:
#                     length = data["w_wm_output_length"]
#                 except:
#                     length = len(tokenizer(data["w_wm_output"])['input_ids'])

#             if length>180 and (data["w_wm_output_z_score"]>thresholds[watermark_type] or data["w_wm_output_prediction"]):
#                 data_list.append(json.loads(line))
#     data_list = data_list[:100]
#     return data_list


# for watermark_type in watermark_types:
#     clean_data_list = read_file(f'/home/jkl6486/sok-llm-watermark/runs_server3/token_200/{watermark_type}/{dataset}/llama/gen_table_w_metrics.jsonl',watermark_type)
    
#     # Define the directory path
#     dir_path = f'/home/jkl6486/sok-llm-watermark/runs_server3/token_200/{watermark_type}/{dataset}/llama/gptattack_base_0'

#     # Create the directory
#     os.makedirs(dir_path, exist_ok=True)

#         # Open the file in write mode
#     with open(f'/home/jkl6486/sok-llm-watermark/runs_server3/token_200/{watermark_type}/{dataset}/llama/gptattack_base_0/gen_table.jsonl', 'w') as f:
#         # Write each item on a new line
#         for item in clean_data_list:
#             f.write(json.dumps(item) + "\n")
            
    
#     src_file = f'/home/jkl6486/sok-llm-watermark/runs_server3/token_200/{watermark_type}/{dataset}/llama/gen_table_meta.json'
#     dst_file = f'/home/jkl6486/sok-llm-watermark/runs_server3/token_200/{watermark_type}/{dataset}/llama/gptattack_base_0/gen_table_meta.json'

#     # Copy the file
#     shutil.copy(src_file, dst_file)
    
    
def read_file(file,watermark_type):
    data_list = []
    with open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
            data['w_wm_output'] = data['w_wm_output_attacked']
            data['w_wm_output_length'] = data['w_wm_output_attacked_length']
            keys_to_remove = [key for key in data if 'w_wm_output_attacked' in key]
            for key in keys_to_remove:
                data.pop(key, None)
            data_list.append(data)
    return data_list

for watermark_type in watermark_types:
    clean_data_list = read_file(f'/home/jkl6486/sok-llm-watermark/runs_server3/token_200/{watermark_type}/{dataset}/llama/gptattack_4/gen_table_attacked.jsonl',watermark_type)
    
    # # Define the directory path
    # dir_path = f'/home/jkl6486/sok-llm-watermark/runs_server3/token_200/{watermark_type}/{dataset}/llama/gptattack_4'

    # # Create the directory
    # os.makedirs(dir_path, exist_ok=True)

        # Open the file in write mode
    with open(f'/home/jkl6486/sok-llm-watermark/runs_server3/token_200/{watermark_type}/{dataset}/llama/gptattack_4/gen_table.jsonl', 'w') as f:
        # Write each item on a new line
        for item in clean_data_list:
            f.write(json.dumps(item) + "\n")
            
    
    src_file = f'/home/jkl6486/sok-llm-watermark/runs_server3/token_200/{watermark_type}/{dataset}/llama/gen_table_meta.json'
    dst_file = f'/home/jkl6486/sok-llm-watermark/runs_server3/token_200/{watermark_type}/{dataset}/llama/gptattack_4/gen_table_meta.json'

    # Copy the file
    shutil.copy(src_file, dst_file)