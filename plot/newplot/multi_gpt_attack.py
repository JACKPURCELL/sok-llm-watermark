import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer
plt.rcParams['font.size'] = 16  # 设置全局字体大小为14

watermark_types = ["john23","xuandong23b","xiaoniu23","rohith23","aiwei23b","scott22","aiwei23"]
thresholds = {
    "john23": 2.358025378986253,
    "xuandong23b": 2.545584412271571,
    "aiwei23": 24.998546600341797,
    "rohith23": 1.8526251316070557,
    "xiaoniu23": 0.00,
    "lean23": 0.984638512134552,
    "scott22": 0.17697394677108003,
    "aiwei23b": 0.2496753585975497
}
attack_times=("gptattack_1" ,"gptattack_2", "gptattack_3","gptattack_4","gptattack_5")
attack_times_mapping = {attack_time: i+1 for i, attack_time in enumerate(attack_times)}  # Add this line

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
fig, ax = plt.subplots(figsize=(7, 7))

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 添加这一行来定义颜色列表
data = []  # Add this line to create a global data list

# Create a nested dictionary to store the data
# Create a nested dictionary to store the data
data_dict = {watermark_type: {attack_time: [] for attack_time in attack_times} for watermark_type in watermark_types}

for i, watermark_type in enumerate(watermark_types):
    for j, attack_time in enumerate(attack_times):
        file_path = f"/home/jkl6486/sok-llm-watermark/runs_server3/token_200/{watermark_type}/c4/llama/{attack_time}/gen_table_w_metrics.jsonl"
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                item['is_greater_than_threshold'] = item['w_wm_output_attacked_z_score'] > thresholds[watermark_type] or item['w_wm_output_attacked_prediction']
                data_dict[watermark_type][attack_time].append(item)  # Append the data to the list

# Create the scatter plot
for watermark_type, data in data_dict.items():
    # x = list(data.keys())
    x = [attack_times_mapping[attack_time] for attack_time in data.keys()]  # Change this line
    y = [sum(item['is_greater_than_threshold'] for item in data[attack_time]) / len(data[attack_time]) for attack_time in attack_times]
    plt.plot(x, y, label=watermark_type)
    plt.scatter(x, y)  # Add this line
# Set the title and labels
plt.title('TPR under Multi ChatGPT 3.5 Paraphase')
plt.xlabel('Attack Times')
plt.ylabel('TPR')

plt.xticks(range(1, 6))  # Add this line

# Set the y-axis range
plt.ylim([0, 1])  # Add this line
# Add a legend
plt.legend(loc='upper right')  # Change this line
plt.tight_layout()  # Add this line

# Show the plot
plt.savefig(f'./plot/newplot/output/multi_gpt_attack.pdf')