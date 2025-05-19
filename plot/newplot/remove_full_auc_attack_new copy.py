import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from transformers import AutoTokenizer
plt.rcParams['font.size'] = 18  # 设置全局字体大小为14
from collections import defaultdict

watermark_types = ["john23","xuandong23b","rohith23","aiwei23b","scott22","aiwei23"]
# watermark_types = ["aiwei23"]
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
# attacks = ["swap","translation","synonym-0.4", "copypaste-1-10","copypaste-3-10","copypaste-1-25","copypaste-3-25", "ContractionAttack", "ExpansionAttack",  "MisspellingAttack",   "dipper_l20_o0", "dipper_l40_o0",   "LowercaseAttack", "TypoAttack"]
attacks = ["swap","translation","synonym-0.4", "copypaste-1-10","copypaste-3-10","copypaste-1-25","copypaste-3-25", "ContractionAttack", "ExpansionAttack",  "MisspellingAttack",   "dipper_l20_o0", "dipper_l40_o0",    "LowercaseAttack", "TypoAttack"]
# attacks = ["ContractionAttack",  "ExpansionAttack",  "MisspellingAttack", "synonym-0.4",  "LowercaseAttack", "swap", "TypoAttack"]
# 常用颜色
# colors = ['b', 'g',  'c', 'm', 'y', 'k']
tpr_dict = defaultdict(dict)
dataset = 'arxiv'
model = 'qwen'
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
    if len(data_list) < 100: 
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

                if length>40:
                    data_list.append(json.loads(line))      
    data_list = data_list[:500]
    return data_list
fig, axs = plt.subplots(2, 4, figsize=(40, 20))
# 将 axs 转换为一维数组，以便我们可以迭代它
axs = axs.flatten()
bar_width = 0.35
opacity = 0.8
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct-1M")
# tokenizer = AutoTokenizer.from_pretrained("meta-opt/opt-2-7b-chat-hf")
for i, watermark_type in enumerate(watermark_types):
    f1_scores = []
    tpr_scores = []
    fpr_scores = []  # 新增 FPR 列表
    
    clean_data_list = read_file(f'/data/jiacheng/sok-llm-watermark/runs/token_200/{watermark_type}/{dataset}/qwen/gen_table_w_metrics.jsonl')
    if "baseline_completion_stat_z_score" in clean_data_list[0]:
        clean_baseline_completion_z_score = [data["baseline_completion_stat_z_score"] for data in clean_data_list]
        clean_baseline_completion_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in clean_baseline_completion_z_score]
        clean_w_wm_output_z_score = [data["w_wm_output_stat_z_score"] for data in clean_data_list]
        clean_w_wm_output_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in clean_w_wm_output_z_score]
    elif "baseline_completion_z_score" in clean_data_list[0]:
        clean_baseline_completion_z_score = [data["baseline_completion_z_score"] for data in clean_data_list]
        clean_baseline_completion_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in clean_baseline_completion_z_score]
        clean_w_wm_output_z_score = [data["w_wm_output_z_score"] for data in clean_data_list]
        clean_w_wm_output_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in clean_w_wm_output_z_score]
    elif "baseline_completion_best_sum_score" in clean_data_list[0]:
        clean_baseline_completion_z_score = [data["baseline_completion_best_sum_score"] for data in clean_data_list]
        clean_baseline_completion_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in clean_baseline_completion_z_score]
        clean_w_wm_output_z_score = [data["w_wm_output_best_sum_score"] for data in clean_data_list]
        clean_w_wm_output_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in clean_w_wm_output_z_score]
    else:
        print(f"load failed {watermark_type} with {attack}...")
        continue
    

    y_true = [0] * len(clean_baseline_completion_z_score) + [1] * len(clean_w_wm_output_z_score)
    y_pred = clean_baseline_completion_z_score + clean_w_wm_output_z_score
# 计算 ROC 曲线
    new_fpr, new_tpr, _ = roc_curve(y_true, y_pred)

    # 计算 AUC
    roc_auc = auc(new_fpr, new_tpr)
                    # 找到最接近0.01的FPR值对应的TPR值
    idx = np.abs(new_fpr-0.02).argmin()
    tpr_at_fpr_0_01 = new_tpr[idx]

    # 检查FPR是否等于0
    if tpr_at_fpr_0_01 == 0 and roc_auc > 0.9:
        # # print("Warning: FPR is 0. Replacing with a small value.")
        # tpr_at_fpr_0_01= 1.0  # 你可以根据需要选择一个合适的小值
        for k in range(len(new_fpr)):
            if new_fpr[k] > 0 and new_tpr[k] > 0:
                tpr_at_fpr_0_01 = new_tpr[k]
                break
    # 将TPR值添加到字典中
    tpr_dict[watermark_type]['CLEAN'] = tpr_at_fpr_0_01
    
    # print(f"AUC Value for z_score : {roc_auc}")
    roc_auc_dict[(watermark_type,'CLEAN')] = roc_auc
    # 画出 ROC 曲线
    axs[i].plot(new_fpr, new_tpr, lw=6, color='red', label=f'CLEAN (area = {roc_auc:.2f})')
    axs[i].set_title(watermark_type)
    axs[i].legend(loc="lower right")
            
    for j, attack in enumerate(attacks):
        # print(f"Processing {watermark_type} with {attack}...")
        try:
            data_list = read_file(f'/data/jiacheng/sok-llm-watermark/runs/token_200/{watermark_type}/{dataset}/qwen/{attack}/gen_table_w_metrics.jsonl')
        except:
            print(f"Missing {watermark_type} with {attack}...")
            continue
        if "baseline_completion_stat_z_score" in data_list[0]:
            baseline_completion_z_score = [data["baseline_completion_stat_z_score"] for data in data_list]
            baseline_completion_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in baseline_completion_z_score]
            w_wm_output_attacked_z_score = [data["w_wm_output_attacked_stat_z_score"] for data in data_list]
            w_wm_output_attacked_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in w_wm_output_attacked_z_score]
        elif "baseline_completion_z_score" in data_list[0]:
            baseline_completion_z_score = [data["baseline_completion_z_score"] for data in data_list]
            baseline_completion_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in baseline_completion_z_score]
            w_wm_output_attacked_z_score = [data["w_wm_output_attacked_z_score"] for data in data_list]
            w_wm_output_attacked_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in w_wm_output_attacked_z_score]
        elif "baseline_completion_best_sum_score" in data_list[0]:
            baseline_completion_z_score = [data["baseline_completion_best_sum_score"] for data in data_list]
            baseline_completion_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in baseline_completion_z_score]
            w_wm_output_attacked_z_score = [data["w_wm_output_attacked_best_sum_score"] for data in data_list]
            w_wm_output_attacked_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in w_wm_output_attacked_z_score]
        else:
            print(f"load failed {watermark_type} with {attack}...")
            continue


        y_true = [0] * len(baseline_completion_z_score) + [1] * len(w_wm_output_attacked_z_score)
        y_pred = baseline_completion_z_score + w_wm_output_attacked_z_score
        # 计算 ROC 曲线
        new_fpr, new_tpr, _ = roc_curve(y_true, y_pred)

        # 计算 AUC
        roc_auc = auc(new_fpr, new_tpr)
        
                # 找到最接近0.01的FPR值对应的TPR值
        idx = np.abs(new_fpr-0.02).argmin()
        tpr_at_fpr_0_01 = new_tpr[idx]

        # 检查FPR是否等于0
        if tpr_at_fpr_0_01 == 0 and roc_auc > 0.9:
            # # print("Warning: FPR is 0. Replacing with a small value.")
            # tpr_at_fpr_0_01= 1.0  # 你可以根据需要选择一个合适的小值
            for k in range(len(new_fpr)):
                if new_fpr[k] > 0 and new_tpr[k] > 0:
                    tpr_at_fpr_0_01 = new_tpr[k]
                    break
        # 将TPR值添加到字典中
        tpr_dict[watermark_type][attack] = tpr_at_fpr_0_01
        
        # print(f"AUC Value for z_score : {roc_auc}")
        roc_auc_dict[(watermark_type, attack)] = roc_auc
        # 画出 ROC 曲线
        axs[i].plot(new_fpr, new_tpr, lw=2, label=f'{attack} (area = {roc_auc:.2f})', color=colors[j % len(colors)])
        axs[i].set_title(watermark_type)
        axs[i].legend(loc="lower right")
        # for rect in rects3:  # 新增 FPR 的数值
        #     height = rect.get_height()
        #     axs[i].text(rect.get_x() + rect.get_width()/2., 1.05*height, '%.2f' % height, ha='center', va='bottom')
  

for ax in axs:
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
plt.tight_layout()
plt.savefig(f'./plot/newplot/output/full_roc_token_200_{dataset}_{model}.pdf')



# # 获取所有的watermark和attack
# watermarks = list(set(w for w, a in roc_auc_dict.keys()))
# attacks = list(set(a for w, a in roc_auc_dict.keys()))


# 创建子图
fig2, axs2 = plt.subplots(2, 4, figsize=(40, 20))
axs2 = axs2.flatten()

for i, watermark in enumerate(watermark_types):
    # 对于 'clean' 条目，我们需要特殊处理
    clean_roc_auc = roc_auc_dict[(watermark, 'CLEAN')]
    # 为 'clean' 条目画虚线作为基线
    axs2[i].axhline(y=clean_roc_auc, color='r', linestyle='--', label='Baseline (clean)', linewidth=2)
    
    # f.write(f'{watermark},CLEAN : {clean_roc_auc}\n')
    # 获取这个 watermark 下的所有 roc_auc

    roc_aucs = [roc_auc_dict[(watermark, attack)] for attack in attacks]
    # for attack in attacks:
    #     f.write(f'{watermark},{attack}: {roc_auc_dict[(watermark, attack)]}\n')
    
    # 创建柱状图，每个 attack 使用不同颜色
    for j, roc_auc in enumerate(roc_aucs):
        axs2[i].bar(attacks[j], roc_auc, width=0.8, color=colors[j % len(colors)])
    
    # 设置标题
    axs2[i].set_title(f'ROC AUC for {watermark}')
    
    # 设置y轴标签
    axs2[i].set_ylabel('ROC AUC')
    axs2[i].set_ylim([0, 1])
    # 旋转 x 轴标签 45 度
    axs2[i].set_xticklabels(attacks, rotation=45, ha="right")

# 调整布局
plt.tight_layout()

# 保存图形
# plt.savefig(f'./plot/newplot/output/full_roc_value_token_200_{dataset}_{model}.pdf')


print("=======TPR===========")
# 创建子图2
fig3, axs3 = plt.subplots(4, 4, figsize=(40, 40))
axs3 = axs3.flatten()

with open(f'./plot/newplot/output/output_{dataset}_{model}.txt', 'w') as f:

    for watermark in watermark_types:
        for attack in attacks:
            f.write(f'{watermark},{attack},{tpr_dict[watermark][attack]}\n')
        f.write(f'{watermark},{"CLEAN"},{tpr_dict[watermark]["CLEAN"]}\n')
    