import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer
plt.rcParams['font.size'] = 14  # 设置全局字体大小为14
watermark_types = ["john23","xuandong23b","aiwei23","lean23","rohith23","aiwei23b","xiaoniu23"]


tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
fig, ax = plt.subplots(figsize=(12, 7))

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 添加这一行来定义颜色列表
tokens = ['token_200','token_100', 'token_50']
linestyles = ['-', '--', '-.']
for i, watermark_type in enumerate(watermark_types):
    file_paths = [
        f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/c4/opt/gen_table_w_metrics.jsonl',
        f'/home/jkl6486/sok-llm-watermark/runs/token_100/{watermark_type}/c4/opt/gen_table_w_metrics.jsonl',
        f'/home/jkl6486/sok-llm-watermark/runs/token_50/{watermark_type}/c4/opt/gen_table_w_metrics.jsonl'
    ]

    for k,file_path in enumerate(file_paths):
        
        data_list = []
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                try:
                    length = data["w_wm_output_length"]
                except:
                    length = len(tokenizer(data["w_wm_output"])['input_ids'])
                length_limit = 30 if 'token_50' in file_path else 150
                length_limit = 70 if 'token_100' in file_path else length_limit
                if length>length_limit:
                    data_list.append(json.loads(line))
        data_list = data_list[:500]
        # 获取预测的概率
        
        if "baseline_completion_z_score" in data.keys():
            baseline_completion_z_score = [data["baseline_completion_z_score"] for data in data_list]
            baseline_completion_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in baseline_completion_z_score]
            w_wm_output_z_score = [data["w_wm_output_z_score"] for data in data_list]
            w_wm_output_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in w_wm_output_z_score]
        else:
            baseline_completion_z_score = [data["baseline_completion_best_sum_score"] for data in data_list]
            baseline_completion_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in baseline_completion_z_score]
            w_wm_output_z_score = [data["w_wm_output_best_sum_score"] for data in data_list]
            w_wm_output_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in w_wm_output_z_score]
        

        # 创建标签
        y_true = [0] * len(baseline_completion_z_score) + [1] * len(w_wm_output_z_score)

        # 合并预测的概率
        y_score = baseline_completion_z_score + w_wm_output_z_score

         # 计算 ROC 曲线
        new_fpr, new_tpr, _ = roc_curve(y_true, y_score)

        # 计算 AUC
        roc_auc = auc(new_fpr, new_tpr)
        # print(file_path)
        # print(f"AUC Value for z_score : {roc_auc}")
        

        # 画出 ROC 曲线
        label = f'{tokens[k]}_{watermark_type} (AUC = {roc_auc:.3f})'
        print(f"{label} ")
        
        linestyle = linestyles[k] 
        # label = f'token_50_{watermark_type} (AUC = {roc_auc:.2f})' if 'token_50' in file_path else f'token200_{watermark_type} (AUC = {roc_auc:.2f})'
        # linestyle = '--' if 'token_50' in file_path else '-'
        ax.plot(new_fpr, new_tpr, lw=2, color=colors[i % len(colors)], linestyle=linestyle, label=label)


ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
# ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(f'./plot/auc_clean_token.pdf')