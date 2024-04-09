import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer
import matplotlib
plt.rcParams['font.size'] = 14
matplotlib.rcParams['font.family'] = 'optima'
matplotlib.rcParams['font.weight'] = 'medium'
from collections import defaultdict

watermark_types = ["lean23"]
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
    "dipper_l20_o0": "DP-20",
    "dipper_l40_o0": "DP-40",
    "dipper_l40_o20": "DP-40-20",
    "dipper_l60_o20": "DP-60-20",
    "translation": "Trans",
}

# attacks = ["ContractionAttack", "copypaste-3-10", "ExpansionAttack",  "MisspellingAttack", "synonym-0.4", "copypaste-1-10", "dipper_l20_o0", "LowercaseAttack", "swap", "TypoAttack"]
# attacks = ["swap","translation","synonym-0.4", "copypaste-1-10","copypaste-3-10","copypaste-1-25","copypaste-3-25", "ContractionAttack", "ExpansionAttack",  "MisspellingAttack",   "dipper_l20_o0", "dipper_l40_o0",      "LowercaseAttack", "TypoAttack"]
attacks = ["synonym-0.4", "MisspellingAttack", "TypoAttack","swap","copypaste-1-10", "copypaste-3-10","copypaste-1-25","copypaste-3-25","ContractionAttack", "ExpansionAttack", "LowercaseAttack","translation","dipper_l20_o0","dipper_l40_o0"]
colors = [
    'darkorange',
    'deepskyblue',
    'limegreen',
    'violet',
    'goldenrod',
    'lightpink',

    'teal'
]
# attacks = ["ContractionAttack",  "ExpansionAttack",  "MisspellingAttack", "synonym-0.4",  "LowercaseAttack", "swap", "TypoAttack"]
# 常用颜色
# colors = ['b', 'g',  'c', 'm', 'y', 'k']

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
# fig, axs = plt.subplots(2, 4, figsize=(40, 20))
# 将 axs 转换为一维数组，以便我们可以迭代它

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
# tokenizer = AutoTokenizer.from_pretrained("meta-opt/opt-2-7b-chat-hf")
# for i, watermark_type in enumerate(watermark_types):
tpr_dict = defaultdict(dict)
fpr_dict = defaultdict(dict)
datasets = ['c4', 'hc3']
models = ['opt', 'llama']
watermark_type = "lean23"    
for model in models:
    for dataset in datasets: 
        clean_data_list = read_file(f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/{dataset}/{model}/gen_table_w_metrics.jsonl')
        if "baseline_completion_prediction" in clean_data_list[0]:
            clean_baseline_completion_prediction = [data["baseline_completion_prediction"] for data in clean_data_list]
            # clean_baseline_completion_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in clean_baseline_completion_z_score]
            clean_w_wm_output_prediction = [data["w_wm_output_prediction"] for data in clean_data_list]
            
            # clean_w_wm_output_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in clean_w_wm_output_z_score]

        FPR = clean_baseline_completion_prediction.count(True) / len(clean_baseline_completion_prediction)
        TPR = clean_w_wm_output_prediction.count(True) / len(clean_w_wm_output_prediction)
        tpr_dict[model][dataset] = TPR
        fpr_dict[model][dataset] = FPR  


combined_dict = defaultdict(dict)
for model in models:
    for dataset in datasets:
        combined_dict[model + '_' + dataset] = {'TPR': tpr_dict[model][dataset], 'FPR': fpr_dict[model][dataset]}

fig, ax = plt.subplots(figsize=(7, 3.5))

# 定义柱子的宽度
width = 0.2

# 为每种模型和数据集的组合创建一个柱状图
for i, (key, value) in enumerate(combined_dict.items()):
    ax.bar(i - width/2, value['TPR'], width=width, color='darkorange')
    ax.bar(i + width/2, value['FPR'], width=width, color='deepskyblue')
    # ax.text(i - width/2, value['TPR'], f"{value['TPR']:.2f}", ha='center', va='bottom')
    # ax.text(i + width/2, value['FPR'], f"{value['FPR']:.2f}", ha='center', va='bottom')

# 设置x轴的标签
ax.set_xticks(range(len(combined_dict)))
ax.set_xticklabels([key.upper() for key in combined_dict.keys()])

# 添加图例
ax.legend(['TPR', 'FPR'])
# 隐藏左、上、右边框
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# 添加虚线标注y轴刻度
# ax.yaxis.grid(True, linestyle='--', linewidth=0.5,zorder=0)
for y in np.arange(0, 1.01, 0.2):
    ax.axhline(y, color='gray', linewidth=0.5, linestyle='--', zorder=0)
ax.set_ylabel('TPR (with FPR = 0.01)')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# 去除y轴的刻度线但保留刻度标签
ax.tick_params(axis='y', length=0)
plt.tight_layout()

plt.savefig(f'./plot/newplot/output/lean_clean.pdf')










tpr_dict = defaultdict(dict)

dataset = 'c4'

for j, attack in enumerate(attacks):
    # print(f"Processing {watermark_type} with {attack}...")
    try:
        data_list = read_file(f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/{dataset}/opt/{attack}/gen_table_w_metrics.jsonl')
    except:
        print(f"Missing {watermark_type} with {attack}...")
        continue
    if "baseline_completion_prediction" in data_list[0]:
        baseline_completion_prediction = [data["baseline_completion_prediction"] for data in data_list]
        # clean_baseline_completion_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in clean_baseline_completion_z_score]
        w_wm_output_prediction = [data["w_wm_output_attacked_prediction"] for data in data_list]
        # clean_w_wm_output_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in clean_w_wm_output_z_score]
    else:
        print(f"load failed {watermark_type} with {attack}...")
        continue
    TPR = w_wm_output_prediction.count(True) / len(w_wm_output_prediction)
    
    tpr_dict[watermark_type][attack] = TPR



print("=======TPR===========")
# 创建子图2

fig, ax = plt.subplots(figsize=(7, 4))

# 定义柱子的宽度
width = 0.6

# 获取'Lean23'的TPR数据
tpr_data = tpr_dict['lean23']

# 创建柱状图
for i, (label, value) in enumerate(tpr_data.items()):
    ax.bar(i, value, width=width, color=colors[i % len(colors)], alpha=1)


# 设置x轴的标签
ax.set_xticks(range(len(tpr_data)))
# ax.set_xticklabels(list(tpr_data.keys()))
ax.set_xticklabels([attack_replace_dict[attack] for attack in tpr_data.keys()])
# 旋转x轴的标签45度
# 隐藏左、上、右边框
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.set_ylabel('TPR (with FPR = 0.01)')

# 添加虚线标注y轴刻度

# ax.yaxis.grid(True, linestyle='--', linewidth=0.5,zorder=0)

for y in np.arange(0, 1.01, 0.2):
    ax.axhline(y, color='gray', linewidth=0.5, linestyle='--', zorder=0)
    
plt.xticks(rotation=90)
for i, rect in enumerate(ax.patches):
    height = rect.get_height()
    # ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')

# 设置x轴的标签
# ax.set_xticks([j for j in attacks])
# ax.set_xticklabels([attack_replace_dict[attack] for attack in attacks])
# ax.set_xticklabels(attacks)


# 添加图例
# ax.legend()

# plt.show()
# with open(f'./plot/newplot/output/output_{dataset}_{model}.txt', 'w') as f:

#     for watermark in watermark_types:
#         for attack in attacks:
#             f.write(f'{watermark},{attack},{tpr_dict[watermark][attack]}\n')
#         f.write(f'{watermark},{"CLEAN"},{tpr_dict[watermark]["CLEAN"]}\n')
    
# for i, attack in enumerate(attacks):

#     roc_aucs = [tpr_dict[watermark][attack] for watermark in watermark_types]

#     # 创建柱状图，每个 attack 使用不同颜色f
#     for j, roc_auc in enumerate(roc_aucs):
#         axs3[i].bar(watermark_types[j], roc_auc, width=0.8, color=colors[j % len(colors)])
    
#     # 设置标题
#     axs3[i].set_title(f'TPR at FPR=0.01 for {attack}')
#     # 设置y轴标签
#     axs3[i].set_ylabel('TPR at FPR=0.01')
#     axs3[i].set_ylim([0, 1])
#     # 旋转 x 轴标签 45 度
#     axs3[i].set_xticklabels(watermark_types, rotation=45, ha="right")

# 调整布局
plt.tight_layout()

# 保存图形
plt.savefig(f'./plot/newplot/output/lean.pdf')


# methods = list(tpr_dict.keys())
# opt_scores = [tpr_dict[method]["OPT"] for method in methods]
# opt2_scores = [tpr_dict[method]["opt2"] for method in methods]

# x = list(range(len(methods)))  # 将range对象转换为列表


# width = 0.3  # 定义柱子的宽度

# # Plot
# fig, ax = plt.subplots(figsize=(7, 4))
# ax.bar([xi - width/2 for xi in x], opt_scores, width=width, label="OPT", align="center")
# ax.bar([xi + width/2 for xi in x], opt2_scores, width=width, label="opt2", align="center")

# ax.set_xlabel("Methods")
# ax.set_ylabel('TPR (with FPR = 0.01)')
# ax.set_title('TPR at FPR=0.01 for each method and model')
# ax.set_xticks(x)
# ax.set_xticklabels(methods, rotation=30, ha="right")
# ax.legend(loc="lower right")

# plt.tight_layout()


# plt.savefig(f'./plot/auc_clean_model_fpr001.pdf')
