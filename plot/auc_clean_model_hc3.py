from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import roc_curve, auc
from matplotlib.patches import Rectangle
from transformers import AutoTokenizer
plt.rcParams['font.size'] = 14
tpr_dict = defaultdict(dict)
dataset = "c4"
watermark_types = ["john23", "xuandong23b", "aiwei23", "rohith23", "xiaoniu23", "lean23", "scott22", "aiwei23b"]
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

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
fig, axs = plt.subplots(1, 2, figsize=(14, 7))  # 创建两个子图

colors = [
    'darkorange',
    'deepskyblue',
    'limegreen',
    'violet',
    'goldenrod',
    'lightpink',
    'slategray',
    'teal'
]

models = ['OPT', 'LLAMA2']
linestyles = ['-', '-']

for i, watermark_type in enumerate(watermark_types):
    file_paths = [
        f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/{dataset}/opt/gen_table_w_metrics.jsonl',
        f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/{dataset}/llama/gen_table_w_metrics.jsonl'
    ]

    for k, file_path in enumerate(file_paths):
        # Your existing code for processing data and generating ROC curves remains unchanged
    
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
        new_fpr, new_tpr, thresholds = roc_curve(y_true, y_score)


        
        
        # 计算 AUC
        roc_auc = auc(new_fpr, new_tpr)
        # print(file_path)
        
        # 找到最接近0.01的FPR值对应的TPR值
        idx = np.abs(new_fpr-0.01).argmin()
        tpr_at_fpr_0_01 = new_tpr[idx]
        print('threshold_',models[k],watermark_type, ": ",thresholds[idx])
        # 检查FPR是否等于0
        if tpr_at_fpr_0_01 == 0:
            # print("Warning: FPR is 0. Replacing with a small value.")
            tpr_at_fpr_0_01= 1.0  # 你可以根据需要选择一个合适的小值
        # 将TPR值添加到字典中
        tpr_dict[watermark_type][models[k]] = tpr_at_fpr_0_01
        label = f'{replace_dict[watermark_type]} (AUC = {roc_auc:.3f})'
        axs[k].plot(new_fpr, new_tpr, lw=2, color=colors[i % len(colors)], linestyle=linestyles[k], label=label)

# 在每个主图上添加放大的子图
zoomed_inset_coords = [0.0, 0.2, 0.8, 1.0]  # [x1, x2, y1, y2] of the zoomed area
for ax in axs:
    # Add the inset showing the zoomed area [0.8, 1.0] for both FPR and TPR
    axins = ax.inset_axes([0.55, 0.45, 0.4, 0.4])
    axins.set_xlim(zoomed_inset_coords[0], zoomed_inset_coords[1])
    axins.set_ylim(zoomed_inset_coords[2], zoomed_inset_coords[3])

    # Iterate over all lines in the main axes to plot them on the inset
    for line in ax.get_lines():
        axins.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), linestyle=line.get_linestyle())

    # Optionally, add a rectangle in the main plot to show the zoomed area
    rect = Rectangle((zoomed_inset_coords[0], zoomed_inset_coords[2]), zoomed_inset_coords[1]-zoomed_inset_coords[0], zoomed_inset_coords[3]-zoomed_inset_coords[2], linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(rect)

    axins.xaxis.set_visible(False)  # Optionally hide x-axis labels
    axins.yaxis.set_visible(False)  # Optionally hide y-axis labels

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(f'./plot/auc_clean_model_{dataset}.pdf')



methods = list(tpr_dict.keys())
opt_scores = [tpr_dict[method]["OPT"] for method in methods]
llama2_scores = [tpr_dict[method]["LLAMA2"] for method in methods]

x = list(range(len(methods)))  # 将range对象转换为列表
methods = [replace_dict.get(method, method) for method in methods]


width = 0.3  # 定义柱子的宽度

# Plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([xi - width/2 for xi in x], opt_scores, width=width, label="OPT", align="center", color='darkorange')
ax.bar([xi + width/2 for xi in x], llama2_scores, width=width, label="LLAMA2", align="center", color='deepskyblue')

ax.set_xlabel("Methods")
ax.set_ylabel('TPR at FPR=0.01')
ax.set_title('TPR at FPR=0.01 for each method and model')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend(loc="lower right")

plt.tight_layout()


plt.savefig(f'./plot/auc_clean_model_fpr001_{dataset}.pdf')