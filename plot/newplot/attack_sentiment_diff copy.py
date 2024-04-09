import matplotlib.pyplot as plt
import numpy as np
import math
import json
from matplotlib.ticker import FuncFormatter

from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer
plt.rcParams['font.size'] = 14
import matplotlib
matplotlib.rcParams['font.family'] = 'optima'
matplotlib.rcParams['font.weight'] = 'medium'
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
attacks = ["synonym-0.4", "MisspellingAttack", "TypoAttack","swap","copypaste-1-10", "copypaste-3-10","copypaste-1-25","copypaste-3-25","ContractionAttack", "ExpansionAttack", "LowercaseAttack","translation","dipper_l20_o0","dipper_l40_o0"]
attack_replace_dict = {
    "synonym-0.4": "Syno",
    "MisspellingAttack": "Missp",
    "TypoAttack": "Typo",
    "swap": "Swap",
    "copypaste-1-10": "CP1-10",
    "copypaste-3-10": "CP3-10",
    "copypaste-1-25": "CP1-25",
    "copypaste-3-25": "CP3-25",
    "ContractionAttack": "Contra",
    "LowercaseAttack": "LowCase",
    "ExpansionAttack": "Expan",
    "translation": "Trans",
    "dipper_l20_o0": "DP-20",
    "dipper_l40_o0": "DP-40",
    "dipper_l40_o20": "DP-40-20",
    "dipper_l60_o0": "DP-60-0",
    "dipper_l60_o20": "DP-60-20",
    "dipper_l60_o40": "DP-60-40",
}
# attacks = ["ContractionAttack",  "ExpansionAttack",  "MisspellingAttack", "synonym-0.4",  "LowercaseAttack", "swap", "TypoAttack"]
# 常用颜色
# colors = ['b', 'g',  'c', 'm', 'y', 'k']
tpr_dict = defaultdict(dict)
dataset = 'c4'
# 创建一个空字典来存储roc_auc值
roc_auc_dict = {}

def read_file(file):
    print(file)
    
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
    if len(data_list) <100: 
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
fig, axs = plt.subplots(2,2, figsize=(7,7))
# 将 axs 转换为一维数组，以便我们可以迭代它
axs = axs.flatten()
bar_width = 0.8
opacity = 1.0
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
# for i, watermark_type in enumerate(watermark_types):

watermark_type = "scott22"
# for j, attack in enumerate(attacks):
#     # print(f"Processing {watermark_type} with {attack}...")
#     try:
#         data_list = read_file(f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/{dataset}/opt/{attack}/gen_table_w_metrics.jsonl')
#     except:
#         print(f"Missing {watermark_type} with {attack}...")
#         continue
metrics = [
"w_wm_output_vs_w_wm_output_attacked_BERTS",
"w_wm_output_vs_w_wm_output_attacked_p_sp",
"w_wm_output_vs_w_wm_output_attacked_WER",
"w_wm_output_vs_w_wm_output_attacked_BLEU1",

]
with open('./plot/newplot/output/attack_sentiment——output.txt', 'w') as f:
    for i, metric in enumerate(metrics):
        averages = {attack: np.mean([data[metric] for data in read_file(f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/{dataset}/opt/{attack}/gen_table_w_metrics.jsonl')]) for attack in attacks}
        # f.write(f'Average {metric.replace("w_wm_output_vs_w_wm_output_attacked_", "")} for Different Attacks:\n')
        for attack in attacks:
            f.write(f'{attack}, {averages[attack]},{metric.replace("w_wm_output_vs_w_wm_output_attacked_", "")}\n')
        f.write('\n')
        

for i, metric in enumerate(metrics):
    averages = {attack: np.mean([data[metric] for data in read_file(f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/{dataset}/opt/{attack}/gen_table_w_metrics.jsonl')]) for attack in attacks}
    bars = axs[i].bar(range(len(attacks)), [averages[attack] for attack in attacks], bar_width, alpha=opacity, color=[colors[j % len(colors)] for j in range(len(attacks))])  # 修改这一行来设置颜色
    # for j, rect in enumerate(bars):
    #     height = rect.get_height()
    #     axs[i].text(rect.get_x() + rect.get_width() / 2, height, '{:.2f}'.format(height), ha='center', va='bottom')
    if i%2==0:
        axs[i].set_ylabel('Average Value')
    axs[i].set_title(f'{metric.replace("w_wm_output_vs_w_wm_output_attacked_", "VS_").upper()} ')
    labels = [attack_replace_dict[attack] for attack in attacks]
    axs[i].set_xticks(range(len(attacks)))
    axs[i].set_xticklabels(labels, rotation=90)
    axs[i].set_yticks(np.arange(0, 1.05, 0.2))
            # 在 y 轴上画一条水平线作为参考线
    for y in np.arange(0, 1.05, 0.2):
        axs[i].axhline(y, color='gray', linewidth=0.5, linestyle='--', zorder=0)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    # axs[i].legend()
    axs[i].tick_params(axis='y', length=0)

plt.tight_layout()
plt.savefig(f'./plot/newplot/output/attack_sentiment_diff_small.pdf')