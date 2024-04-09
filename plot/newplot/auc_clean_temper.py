from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer
# plt.rcParams['font.size'] = 14  # 设置全局字体大小为14
# watermark_types = ["john23","xuandong23b","aiwei23","lean23","rohith23","aiwei23b","xiaoniu23"]
import matplotlib
plt.rcParams['font.size'] = 14
matplotlib.rcParams['font.family'] = 'optima'
matplotlib.rcParams['font.weight'] = 'medium'
watermark_types = ["john23", "xuandong23b", "xiaoniu23",  "aiwei23","aiwei23b"]


tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
fig, ax = plt.subplots(figsize=(7, 4))
tpr_dict = defaultdict(dict)

replace_dict = {
    "john23": "TGRL",
    "xuandong23b": "UG",
    "aiwei23": "UPV",
    "rohith23": "RDF",
    "xiaoniu23": "UB",
    "scott22": "GO",
    "aiwei23b": "SIR",
}

# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 添加这一行来定义颜色列表
watermark_colors = {
    "rohith23": "orange",
    "xuandong23b": "deepskyblue",
    "john23": "limegreen",
    "aiwei23": "purple",
    "xiaoniu23": "magenta",
    "aiwei23b": "red",
    "scott22": "royalblue"
}

labels = ['0.7', '1.0','1.3']
linestyles = ['-', '--', '-.']
for i, watermark_type in enumerate(watermark_types):
    file_paths = [
         f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/c4/opt-temp0.7/gen_table_w_metrics.jsonl',  # 添加新的文件路径
        f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/c4/opt-temp1.0/gen_table_w_metrics.jsonl',
        f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/c4/opt-temp1.3/gen_table_w_metrics.jsonl',
    ]

    for k,file_path in enumerate(file_paths):
        
        data_list = []
        try:
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
        except:
            continue
        # 获取预测的概率
        try:
            baseline_completion_z_score = [data["baseline_completion_z_score"] for data in data_list]
            baseline_completion_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in baseline_completion_z_score]
            w_wm_output_z_score = [data["w_wm_output_z_score"] for data in data_list]
            w_wm_output_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in w_wm_output_z_score]
        except:
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
        
        # 找到最接近0.01的FPR值对应的TPR值
        idx = np.abs(new_fpr-0.01).argmin()
        tpr_at_fpr_0_01 = new_tpr[idx]
        # print('threshold_',models[k],watermark_type, ": ",thresholds[idx])
        # 检查FPR是否等于0
        if tpr_at_fpr_0_01 == 0:
            # print("Warning: FPR is 0. Replacing with a small value.")
            tpr_at_fpr_0_01= 1.0  # 你可以根据需要选择一个合适的小值
        # 将TPR值添加到字典中
        tpr_dict[watermark_type][labels[k]] = tpr_at_fpr_0_01
        
        # print(file_path)
        print(f"AUC Value for z_score : {roc_auc}")

        # 画出 ROC 曲线
        label = f'{labels[k]}_{replace_dict[watermark_type]} ({roc_auc:.3f})' 
        # linestyle = '--' if 'token_50' in file_path else '-'
        ax.plot(new_fpr, new_tpr, lw=2, color=watermark_colors[watermark_type], linestyle=linestyles[k], label=label)


ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
# ax.legend(loc="lower right")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, columnspacing=0.5)
plt.tight_layout()
plt.savefig(f'./plot/newplot/output/temper_auc_cleantoken.pdf')


fig, ax = plt.subplots(figsize=(7, 4))

methods = list(tpr_dict.keys())
scores_07 = [tpr_dict[method]["0.7"] for method in methods]
scores_10 = [tpr_dict[method]["1.0"] for method in methods]
scores_13 = [tpr_dict[method]["1.3"] for method in methods]

x = list(range(len(methods)))  # 将range对象转换为列表
methods = [replace_dict.get(method, method) for method in methods]

width = 0.2  # 定义柱子的宽度
gap = 0.00  # 定义柱子之间的间隙

# Plot
ax.bar([xi - width - gap for xi in x], scores_07, width=width, label="0.7", color='darkorange')
ax.bar(x, scores_10, width=width, label="1.0", color='deepskyblue')
ax.bar([xi + width + gap for xi in x], scores_13, width=width, label="1.3", color='limegreen')

for y in np.arange(0, 1.05, 0.2):
    ax.axhline(y, color='gray', linewidth=0.5, linestyle='--', zorder=0)
            
            
ax.set_xlabel("Methods")
ax.set_ylabel('TPR (with FPR = 0.01)')
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


plt.savefig(f'./plot/newplot/output/temper_auc_clean_model_fpr001.pdf')
print(f'./plot/newplot/output/temper_auc_clean_model_fpr001.pdf')