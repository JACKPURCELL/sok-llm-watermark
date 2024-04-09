import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer
import matplotlib
plt.rcParams['font.size'] = 14

matplotlib.rcParams['font.family'] = 'optima'
matplotlib.rcParams['font.weight'] = 'medium'
# watermark_types = ["rohith23", "xuandong23b", "john23", "aiwei23", "xiaoniu23", "aiwei23b", "scott22"]
watermark_types = ["john23", "xuandong23b",  "rohith23", "scott22", "xiaoniu23",  "aiwei23","aiwei23b"]

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

watermark_colors = {
    "rohith23": "orange",
    "xuandong23b": "deepskyblue",
    "john23": "limegreen",
    "aiwei23": "purple",
    "xiaoniu23": "magenta",
    "aiwei23b": "red",
    "scott22": "royalblue"
}


tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# fig, ax = plt.subplots(figsize=(7, 7))

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 添加这一行来定义颜色列表
# fig, axs = plt.subplots(2, 4, figsize=(28, 14))
keys_to_include = ["w_wm_output_repetition_2", "w_wm_output_repetition_3", "w_wm_output_repetition_4", "w_wm_output_diversity",  "w_wm_output_coherence"]
# "w_wm_output_log_diversity",
wm_avg_var_dict_list = []
for i, watermark_type in enumerate(watermark_types):
    data_dict = {}
    
    file_path = f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/c4/opt/gen_table_w_metrics.jsonl'
    data_list = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            try:
                length = data["w_wm_output_length"]
            except:
                length = len(tokenizer(data["w_wm_output"])['input_ids'])
            length_limit = 30 if 'token_50' in file_path else 150
            
            if length>length_limit:
                data_list.append(json.loads(line))
    data_list = data_list[:500]
    # 获取预测的概率
            
    for data in data_list:
        for key in data.keys():
            if 'no_wm_output_vs_w_wm_output' in key or key in keys_to_include:
                clip_key = key.replace('no_wm_output_vs_w_wm_output_', 'VS_')
                clip_key = clip_key.replace('w_wm_output_', '')
                if clip_key not in data_dict:
                    data_dict[clip_key] = []
                data_dict[clip_key].append(data[key])

    # 计算每个键的平均值和方差
    # avg_var_dict = {key: (np.mean(values), np.var(values)) for key, values in data_dict.items()}
    avg_var_dict = {key: (np.mean([v for v in values if not np.isnan(v)]), np.var([v for v in values if not np.isnan(v)])) for key, values in data_dict.items()}
    avg_var_dict_new = {}
    
    avg_var_dict_new["VS_BERTS"] = avg_var_dict["VS_BERTS"]
    avg_var_dict_new["VS_MAUVE"] = avg_var_dict["VS_mauve"]
    
    avg_var_dict_new["VS_P_SP"] = avg_var_dict["VS_p_sp"]
    # avg_var_dict_new["VS_"] = avg_var_dict["coherence"]
    wm_avg_var_dict_list.append(avg_var_dict_new)
    
    
fig, ax = plt.subplots(figsize=(7, 4))
VS_BERTS = [avg_var_dict_new["VS_BERTS"][0] for avg_var_dict_new in wm_avg_var_dict_list]
VS_MAUVE = [avg_var_dict_new["VS_MAUVE"][0] for avg_var_dict_new in wm_avg_var_dict_list]
VS_P_SP = [avg_var_dict_new["VS_P_SP"][0] for avg_var_dict_new in wm_avg_var_dict_list]

VS_BERTS_err = [avg_var_dict_new["VS_BERTS"][1] for avg_var_dict_new in wm_avg_var_dict_list]
VS_MAUVE_err  = [avg_var_dict_new["VS_MAUVE"][1] for avg_var_dict_new in wm_avg_var_dict_list]
VS_P_SP_err  = [avg_var_dict_new["VS_P_SP"][1] for avg_var_dict_new in wm_avg_var_dict_list]
x = list(range(len(watermark_types)))  # 将range对象转换为列表
methods = [replace_dict.get(method, method) for method in watermark_types]

width = 0.25  # 定义柱子的宽度
gap = 0.0  # 定义柱子之间的间隙

# Plot
ax.bar([xi - width - gap for xi in x], VS_BERTS, width=width, yerr=VS_BERTS_err, label="VS_BERTS", color='darkorange')
ax.bar(x, VS_MAUVE, width=width, yerr=VS_MAUVE_err, label="VS_MAUVE", color='deepskyblue')
ax.bar([xi + width + gap for xi in x], VS_P_SP, width=width, yerr=VS_P_SP_err, label="VS_P_SP", color='limegreen')

for y in np.arange(0, 1.05, 0.2):
    ax.axhline(y, color='gray', linewidth=0.5, linestyle='--', zorder=0)
            
# ax.set_xlabel("Methods")
ax.set_ylabel('Average Value')
# ax.set_title('TPR at FPR=0.01 for each method and temperature')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(loc="lower right")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# 去除y轴的刻度线但保留刻度标签
ax.tick_params(axis='y', length=0)
plt.tight_layout()


plt.savefig(f'./plot/newplot/output/clean_sentiment_new_small.pdf')