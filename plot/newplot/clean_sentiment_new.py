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
watermark_types = ["rohith23", "xuandong23b", "john23", "aiwei23", "xiaoniu23", "aiwei23b", "scott22"]

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
data_dict = {}
# fig, axs = plt.subplots(2, 4, figsize=(28, 14))
keys_to_include = ["w_wm_output_repetition_2", "w_wm_output_repetition_3", "w_wm_output_repetition_4", "w_wm_output_diversity",  "w_wm_output_coherence"]
# "w_wm_output_log_diversity",
wm_avg_var_dict_list = []
for i, watermark_type in enumerate(watermark_types):
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
    
fig, axs = plt.subplots(3, 1, figsize=(7, 8))
plt.subplots_adjust(hspace=0.0)  # 调整子图之间的间距
axs = axs.flatten()
labels = [replace_dict[watermark_type] for watermark_type in watermark_types]
# labels[1] = "xd23b"
for i, key in enumerate(wm_avg_var_dict_list[0].keys()):
    for k,watermark_type in enumerate(watermark_types):
        # 获取这个watermark_type的数据
        avg, var = wm_avg_var_dict_list[k][key]

        axs[i].bar(watermark_type, avg, yerr=var, label=watermark_type,color=watermark_colors[watermark_type], width=0.4)
    
    # 设置子图的标题和图例
    axs[i].set_title(key)
    # axs[i].legend()

    axs[i].yaxis.grid(False)

    if i == 2:
        axs[i].set_xticklabels(labels)
    else:
        axs[i].set_xticklabels([])

    axs[i].set_ylabel('Values')
      # 设置 y 轴的刻度
    if i !=2: 
        axs[i].set_yticks(np.arange(0, 1.0, 0.2))
            # 在 y 轴上画一条水平线作为参考线
        for y in np.arange(0, 1.0, 0.2):
            axs[i].axhline(y, color='gray', linewidth=0.5, linestyle='-')
    else:
        axs[i].set_yticks(np.arange(0, 0.5, 0.1))
            # 在 y 轴上画一条水平线作为参考线
        for y in np.arange(0,  0.5, 0.1):
            axs[i].axhline(y, color='gray', linewidth=0.5, linestyle='-')
    

    
     # 去除上下右的框线
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    

    
    
plt.tight_layout()

plt.savefig(f'./plot/newplot/output/clean_sentiment_new_small.pdf')