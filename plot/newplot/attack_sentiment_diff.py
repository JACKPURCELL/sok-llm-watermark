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
attacks = ["swap","translation","synonym-0.4", "ContractionAttack", "ExpansionAttack",  "MisspellingAttack",   "dipper_l20_o0", "dipper_l40_o0", "LowercaseAttack", "TypoAttack"]
# attacks = ["ContractionAttack",  "ExpansionAttack",  "MisspellingAttack", "synonym-0.4",  "LowercaseAttack", "swap", "TypoAttack"]
# 常用颜色
# colors = ['b', 'g',  'c', 'm', 'y', 'k']
tpr_dict = defaultdict(dict)
dataset = 'c4'
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
fig, axs = plt.subplots(3,3, figsize=(20, 20))
# 将 axs 转换为一维数组，以便我们可以迭代它
axs = axs.flatten()
bar_width = 0.8
opacity = 0.8
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
# for i, watermark_type in enumerate(watermark_types):

watermark_type = "xiaoniu23"
# for j, attack in enumerate(attacks):
#     # print(f"Processing {watermark_type} with {attack}...")
#     try:
#         data_list = read_file(f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/{dataset}/opt/{attack}/gen_table_w_metrics.jsonl')
#     except:
#         print(f"Missing {watermark_type} with {attack}...")
#         continue
metrics = [
"w_wm_output_vs_w_wm_output_attacked_BERTS",
"w_wm_output_vs_w_wm_output_attacked_WER",
"w_wm_output_vs_w_wm_output_attacked_BLEU",
"w_wm_output_vs_w_wm_output_attacked_BLEU1",
"w_wm_output_vs_w_wm_output_attacked_BLEU2",
"w_wm_output_vs_w_wm_output_attacked_BLEU3",
"w_wm_output_vs_w_wm_output_attacked_BLEU_4",
"w_wm_output_vs_w_wm_output_attacked_p_sp",
"w_wm_output_vs_w_wm_output_attacked_mauve"
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
    # axs[i].bar(range(len(attacks)), [averages[attack] for attack in attacks], bar_width, alpha=opacity)
    axs[i].bar(range(len(attacks)), [averages[attack] for attack in attacks], bar_width, alpha=opacity, color=[colors[j % len(colors)] for j in range(len(attacks))])  # 修改这一行来设置颜色
    axs[i].set_xlabel('Attacks')
    axs[i].set_ylabel('Average Value')
    axs[i].set_title(f'Average {metric.replace("w_wm_output_vs_w_wm_output_attacked_", "")} for Different Attacks')
    axs[i].set_xticks(range(len(attacks)))
    axs[i].set_xticklabels(attacks, rotation=45)
    axs[i].legend()

plt.tight_layout()
plt.savefig(f'./plot/newplot/output/attack_sentiment_diff.pdf')