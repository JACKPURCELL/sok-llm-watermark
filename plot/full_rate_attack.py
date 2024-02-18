import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import f1_score, confusion_matrix
from transformers import AutoTokenizer
plt.rcParams['font.size'] = 18  # 设置全局字体大小为14

watermark_types = ["john23","xuandong23b","aiwei23","lean23","rohith23","xiaoniu23"]
# attacks = ["ContractionAttack", "copypaste-3-10", "ExpansionAttack",  "MisspellingAttack", "synonym-0.4", "copypaste-1-10", "dipper_l20_o0", "LowercaseAttack", "swap", "TypoAttack"]
attacks = ["swap","synonym-0.4", "copypaste-1-10","copypaste-3-10", "ContractionAttack", "ExpansionAttack",  "MisspellingAttack",   "dipper_l20_o0", "LowercaseAttack", "TypoAttack"]
# attacks = ["ContractionAttack",  "ExpansionAttack",  "MisspellingAttack", "synonym-0.4",  "LowercaseAttack", "swap", "TypoAttack"]
# 常用颜色
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']



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
            data_list = data_list[:500]
    return data_list
fig, axs = plt.subplots(2, 3, figsize=(40, 20))
# 将 axs 转换为一维数组，以便我们可以迭代它
axs = axs.flatten()
bar_width = 0.35
opacity = 0.8
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
for i, watermark_type in enumerate(watermark_types):
    f1_scores = []
    tpr_scores = []
    fpr_scores = []  # 新增 FPR 列表
    
    clean_data_list = read_file(f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/c4/llama/gen_table_w_metrics.jsonl')
    try:
        clean_baseline_completion_prediction = [1 if data["baseline_completion_prediction"] else 0 for data in clean_data_list]
        clean_w_wm_output_attacked_prediction = [1 if data["w_wm_output_prediction"] else 0 for data in clean_data_list]
    except:
        clean_baseline_completion_prediction = [1 if data["baseline_completion_watermarked"] else 0 for data in clean_data_list]
        clean_w_wm_output_attacked_prediction = [1 if data["w_wm_output_watermarked"] else 0 for data in clean_data_list]
    
    y_true = [0] * len(clean_baseline_completion_prediction) + [1] * len(clean_w_wm_output_attacked_prediction)
    y_pred = clean_baseline_completion_prediction + clean_w_wm_output_attacked_prediction
# 计算 F1 分数
    f1 = f1_score(y_true, y_pred)
    f1_scores.append(f1)

    # 计算 TPR
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)  # FPR 是 FP/(FP+TN)
    
    axs[i].axhline(f1, color='r', linestyle='--', label='Baseline F1', linewidth=2)
    axs[i].axhline(tpr, color='b', linestyle='--', label='Baseline TPR', linewidth=2)
    axs[i].axhline(fpr, color='g', linestyle='--', label='Baseline FPR', linewidth=2)
            
    for j, attack in enumerate(attacks):
        print(f"Processing {watermark_type} with {attack}...")
        try:
            data_list = read_file(f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/c4/llama/{attack}/gen_table_w_metrics.jsonl')
        except:
            print(f"Missing {watermark_type} with {attack}...")
            continue
        try:
            baseline_completion_prediction = [1 if data["baseline_completion_prediction"] else 0 for data in data_list]
            w_wm_output_attacked_prediction = [1 if data["w_wm_output_attacked_prediction"] else 0 for data in data_list]
        except:
            baseline_completion_prediction = [1 if data["baseline_completion_watermarked"] else 0 for data in data_list]
            w_wm_output_attacked_prediction = [1 if data["w_wm_output_attacked_watermarked"] else 0 for data in data_list]


        y_true = [0] * len(baseline_completion_prediction) + [1] * len(w_wm_output_attacked_prediction)
        y_pred = baseline_completion_prediction + w_wm_output_attacked_prediction
    # 计算 F1 分数
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

        # 计算 TPR
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn)
        tpr_scores.append(tpr)
        fpr = fp / (fp + tn)  # FPR 是 FP/(FP+TN)
        fpr_scores.append(fpr)  # 将 FPR 添加到列表中


       # 画出 F1 分数和 TPR 的柱状图
        rects1 = axs[i].bar(j - bar_width/2, f1, bar_width, color=colors[j % len(colors)], label=f'{attack} F1')
        rects2 = axs[i].bar(j + bar_width/2, tpr, bar_width, color=colors[j % len(colors)], alpha=0.5, label=f'{attack} TPR')
        # rects1 = axs[i].bar(j - bar_width, f1, bar_width, color=colors[j % len(colors)], label=f'{attack} F1')
        # rects2 = axs[i].bar(j, tpr, bar_width, color=colors[j % len(colors)], alpha=0.7, label=f'{attack} TPR')
        # rects3 = axs[i].bar(j + bar_width, fpr, bar_width, color=colors[j % len(colors)], alpha=0.5, label=f'{attack} FPR')  # 新增 FPR 的柱状图
        # 在柱状图上添加数值
        for rect in rects1:
            height = rect.get_height()
            axs[i].text(rect.get_x() + rect.get_width()/2., 1.05*height, '%.2f' % height, ha='center', va='bottom')

        for rect in rects2:
            height = rect.get_height()
            axs[i].text(rect.get_x() + rect.get_width()/2., 1.05*height, '%.2f' % height, ha='center', va='bottom')

        # for rect in rects3:  # 新增 FPR 的数值
        #     height = rect.get_height()
        #     axs[i].text(rect.get_x() + rect.get_width()/2., 1.05*height, '%.2f' % height, ha='center', va='bottom')
    axs[i].set_ylim([0, 1])
    axs[i].set_title(watermark_type)
    axs[i].set_xticks(range(len(attacks)))
    axs[i].set_xticklabels(attacks, rotation=45)
    # axs[i].legend()

plt.tight_layout()
plt.savefig(f'./plot/f1_scores_and_tpr_token_200_llama.pdf')