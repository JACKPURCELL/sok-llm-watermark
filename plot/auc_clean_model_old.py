from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer
import matplotlib.cm as cm
plt.rcParams['font.size'] = 14  # 设置全局字体大小为14
# watermark_types = ["john23","xuandong23b","aiwei23","lean23","rohith23","aiwei23b"]
watermark_types = ["john23","xuandong23b","aiwei23","rohith23","aiwei23b","xiaoniu23","lean23"]


tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
fig, ax = plt.subplots(figsize=(7, 7))
tpr_dict = defaultdict(dict)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 添加这一行来定义颜色列表
# colors = cm.rainbow(np.linspace(0, 1, len(watermark_types)))  # 生成一个颜色映射
models = ['OPT']
alphas = [1.0, 0.7]  # 定义透明度列表，你可以根据需要调整这个列表

linestyles = ['-', '--', '-.']
for i, watermark_type in enumerate(watermark_types):
    file_paths = [
        f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/c4/opt/gen_table_w_metrics.jsonl',
        # f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/c4/llama/gen_table_w_metrics.jsonl'
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
        
        # 找到最接近0.01的FPR值对应的TPR值
        idx = np.abs(new_fpr-0.01).argmin()
        tpr_at_fpr_0_01 = new_tpr[idx]

        # 检查FPR是否等于0
        if tpr_at_fpr_0_01 == 0:
            # print("Warning: FPR is 0. Replacing with a small value.")
            tpr_at_fpr_0_01= 1.0  # 你可以根据需要选择一个合适的小值
        # 将TPR值添加到字典中
        tpr_dict[watermark_type][models[k]] = tpr_at_fpr_0_01
        
        # 画出 ROC 曲线
        label = f'{models[k]}_{watermark_type} (AUC = {roc_auc:.3f})'
        # print(f"{label} ")
        print(f"{models[k]}_{watermark_type} : {tpr_at_fpr_0_01:.3f}")
        
        linestyle = linestyles[k] 
        # label = f'token_50_{watermark_type} (AUC = {roc_auc:.2f})' if 'token_50' in file_path else f'token200_{watermark_type} (AUC = {roc_auc:.2f})'
        # linestyle = '--' if 'token_50' in file_path else '-'
        ax.plot(new_fpr, new_tpr, lw=2, color=colors[i % len(colors)], linestyle=linestyle, label=label)


ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
# ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(f'./plot/auc_clean_model.pdf')




methods = list(tpr_dict.keys())
opt_scores = [tpr_dict[method]["OPT"] for method in methods]
# llama2_scores = [tpr_dict[method]["LLAMA2"] for method in methods]

x = list(range(len(methods)))  # 将range对象转换为列表


width = 0.3  # 定义柱子的宽度

# Plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([xi - width/2 for xi in x], opt_scores, width=width, label="OPT", align="center")
# ax.bar([xi + width/2 for xi in x], llama2_scores, width=width, label="LLAMA2", align="center")

ax.set_xlabel("Methods")
ax.set_ylabel('TPR at FPR=0.01')
ax.set_title('TPR at FPR=0.01 for each method and model')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=30, ha="right")
ax.legend(loc="lower right")

plt.tight_layout()


plt.savefig(f'./plot/auc_clean_model_fpr001.pdf')