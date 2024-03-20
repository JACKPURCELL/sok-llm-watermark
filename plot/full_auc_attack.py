import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer
watermark_types = ["john23","xuandong23b","aiwei23","lean23","rohith23","xiaoniu23"]
attacks = ["ContractionAttack", "copypaste-3-10", "ExpansionAttack",  "MisspellingAttack", "synonym-0.4", "copypaste-1-10", "dipper_l20_o0", "LowercaseAttack", "swap", "TypoAttack"]

fig, axs = plt.subplots(2, 3, figsize=(40, 20))

# 将 axs 转换为一维数组，以便我们可以迭代它
axs = axs.flatten()
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
for i, watermark_type in enumerate(watermark_types):
    for attack in attacks:
        print(f"Processing {watermark_type} with {attack}...")
        file = f'/home/ljc/sok-llm-watermark/runs/{watermark_type}/c4/opt/{attack}/gen_table_w_metrics.jsonl'
        data_list = []
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line)
                try:
                    length = data["w_wm_output_attacked_length"]
                except:
                    length = len(tokenizer(data["w_wm_output_attacked"])['input_ids'])
                if length>100:
                    data_list.append(json.loads(line))
        data_list = data_list[:500]
        # 获取预测的概率
        try:
            baseline_completion_z_score = [data["baseline_completion_z_score"] for data in data_list]
            baseline_completion_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in baseline_completion_z_score]
            w_wm_output_z_score = [data["w_wm_output_attacked_z_score"] for data in data_list]
            w_wm_output_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in w_wm_output_z_score]
        except:
            continue

        # 创建标签
        y_true = [0] * len(baseline_completion_z_score) + [1] * len(w_wm_output_z_score)

        # 合并预测的概率
        y_score = baseline_completion_z_score + w_wm_output_z_score

        # 计算 ROC 曲线
        new_fpr, new_tpr, _ = roc_curve(y_true, y_score)

        # 计算 AUC
        roc_auc = auc(new_fpr, new_tpr)
        print(f"AUC Value for z_score : {roc_auc}")

        # 画出 ROC 曲线
        axs[i].plot(new_fpr, new_tpr, lw=2, label=f'{attack} (area = {roc_auc:.2f})')
        axs[i].set_title(watermark_type)
        axs[i].legend(loc="lower right")



for ax in axs:
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

plt.tight_layout()
plt.savefig(f'./plot/roc_curve_entry2.pdf')