import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer
plt.rcParams['font.size'] = 14  # 设置全局字体大小为14
watermark_types = ["john23","xuandong23b","aiwei23","lean23","rohith23","aiwei23b","xiaoniu23","kiyoon23"]


# tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# fig, ax = plt.subplots(figsize=(7, 7))

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 添加这一行来定义颜色列表
data_dict = {}
fig, axs = plt.subplots(2, 4, figsize=(28, 14))
keys_to_include = ["w_wm_output_repetition_2", "w_wm_output_repetition_3", "w_wm_output_repetition_4", "w_wm_output_diversity",  "w_wm_output_coherence"]
# "w_wm_output_log_diversity",
for i, watermark_type in enumerate(watermark_types):
    file_path = f'~/sok-llm-watermark/runs/token_200/{watermark_type}/c4/llama/gen_table_w_metrics.jsonl'
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

    # 画柱状图
    labels = list(avg_var_dict.keys())
    means = [avg for avg, var in avg_var_dict.values()]
    variances = [var for avg, var in avg_var_dict.values()]

    x = np.arange(len(labels))  # 标签的位置

    row = i // 4
    col = i % 4
    axs[row, col].bar(x, means, yerr=variances, align='center', alpha=0.5, ecolor='black', capsize=10, label='Mean with variance')
    axs[row, col].set_ylabel('Values')
    axs[row, col].set_xticks(x)
    axs[row, col].set_xticklabels(labels, rotation=45)
    axs[row, col].set_title(f'Mean and Variance of each key for {watermark_type}')
    axs[row, col].yaxis.grid(True)
    axs[row, col].legend()

plt.tight_layout()
plt.savefig(f'./plot/clean_sentiment_llama.pdf')