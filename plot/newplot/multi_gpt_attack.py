import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer
import seaborn as sns
import matplotlib

plt.rcParams['font.size'] = 14  # 设置全局字体大小为14
matplotlib.rcParams['font.family'] = 'optima'
matplotlib.rcParams['font.weight'] = 'medium'
watermark_types = ["john23","xuandong23b","xiaoniu23","rohith23","aiwei23b","scott22","aiwei23"]
#opt
# thresholds = {
#     "john23": 2.358025378986253,
#     "xuandong23b": 2.545584412271571,
#     "aiwei23": 24.998546600341797,
#     "rohith23": 1.8526251316070557,
#     "xiaoniu23": 0.00,
#     "lean23": 0.984638512134552,
#     "scott22": 0.17697394677108003,
#     "aiwei23b": 0.2496753585975497
# }

#llama
thresholds = {
    "john23": 2.553769592276246,
    "xuandong23b": 2.3276405323333744,
    "aiwei23": 24.984704971313477,
    "rohith23": 1.9777930974960327,
    "xiaoniu23": 3.460620403289795,
    "lean23": 0.9477958083152771,
    "scott22": 0.2150683972747871,
    "aiwei23b": 0.21475310075384416
}
watermark_colors = {
    "rohith23": "orange",
    "xuandong23b": "deepskyblue",
    "john23": "limegreen",
    "aiwei23": "purple",
    "xiaoniu23": "magenta",
    "aiwei23b": "red",
    "scott22": "royalblue"
}
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

# threshold_ OPT john23 :  2.358025378986253
# threshold_ LLAMA2 john23 :  2.553769592276246
# threshold_ OPT xuandong23b :  2.545584412271571
# threshold_ LLAMA2 xuandong23b :  2.3276405323333744
# threshold_ OPT aiwei23 :  24.998546600341797
# threshold_ LLAMA2 aiwei23 :  24.984704971313477
# threshold_ OPT rohith23 :  1.8526251316070557
# threshold_ LLAMA2 rohith23 :  1.9777930974960327
# threshold_ OPT xiaoniu23 :  313.2385559082031
# threshold_ LLAMA2 xiaoniu23 :  3.460620403289795
# threshold_ OPT lean23 :  0.984638512134552
# threshold_ LLAMA2 lean23 :  0.9477958083152771
# threshold_ OPT scott22 :  0.17697394677108003
# threshold_ LLAMA2 scott22 :  0.2150683972747871
# threshold_ OPT aiwei23b :  0.2496753585975497
# threshold_ LLAMA2 aiwei23b :  0.21475310075384416
attack_times=("gptattack_1" ,"gptattack_2", "gptattack_3","gptattack_4","gptattack_5")

attack_times_2=("exp_3_gptattack_1" ,"exp_3_gptattack_2", "exp_3_gptattack_3","exp_3_gptattack_4","exp_3_gptattack_5")
attack_times_mapping = {attack_time: i+1 for i, attack_time in enumerate(attack_times)}  # Add this line
attack_times_mapping_2 = {attack_time: i+1 for i, attack_time in enumerate(attack_times_2)}  # Add this line

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
fig, ax = plt.subplots(figsize=(7, 4))

# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 添加这一行来定义颜色列表
data = []  # Add this line to create a global data list

# Create a nested dictionary to store the data
# Create a nested dictionary to store the data
data_dict = {watermark_type: {attack_time: [] for attack_time in attack_times} for watermark_type in watermark_types}
data_dict_2 = {watermark_type: {attack_time: [] for attack_time in attack_times_2} for watermark_type in watermark_types}

for i, watermark_type in enumerate(watermark_types):
    for j, attack_time in enumerate(attack_times):
        file_path = f"/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/c4/llama/{attack_time}/gen_table_w_metrics.jsonl"
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                item['is_greater_than_threshold'] = item['w_wm_output_attacked_z_score'] > thresholds[watermark_type] 
                data_dict[watermark_type][attack_time].append(item)  # Append the data to the list

for i, watermark_type in enumerate(watermark_types):
    for j, attack_time in enumerate(attack_times_2):
        file_path = f"/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/c4/llama/{attack_time}/gen_table_w_metrics.jsonl"
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                item['is_greater_than_threshold'] = item['w_wm_output_attacked_z_score'] > thresholds[watermark_type]
                data_dict_2[watermark_type][attack_time].append(item)  # Append the data to the list
    
    import numpy as np
                
# Create the scatter plot
for (watermark_type1, data1), (watermark_type2, data2) in zip(data_dict.items(), data_dict_2.items()):
    x = [attack_times_mapping[attack_time] for attack_time in data1.keys()]  # Change this line
    y = [sum(item['is_greater_than_threshold'] for item in data1[attack_time]) / len(data1[attack_time]) for attack_time in attack_times]
    y2 = [sum(item['is_greater_than_threshold'] for item in data2[attack_time]) / len(data2[attack_time]) for attack_time in attack_times_2]

    y_means1 = [np.mean([val1, val2]) for val1, val2 in zip(y, y2)]
    y_stds1 = [np.std([val1, val2]) for val1, val2 in zip(y, y2)]
    import numpy as np

    # 将列表转换为numpy数组
    y_means1 = np.array(y_means1)
    y_stds1 = np.array(y_stds1)
    # plt.errorbar(x, y_means1, yerr=y_stds1, label=replace_dict[watermark_type1], fmt='-o',color=watermark_colors[watermark_type1])  # Use errorbar here
    plt.plot(x, y_means1, '-o', color=watermark_colors[watermark_type1], label=replace_dict[watermark_type1])
    plt.fill_between(x, y_means1 - y_stds1, y_means1 + y_stds1, color=watermark_colors[watermark_type1], alpha=0.2)

    # sns.lineplot(x=x, y=y_means1, err=y_stds1, label=replace_dict[watermark_type1], color=watermark_colors[watermark_type1], linewidth=2.5,err_style='band')
    # fig.lineplot(sub_df['Epochs']-1, sub_df['Fidelity_mean'], err=sub_df['Fidelity_std'], err_style='band', label=optimizer, color=colors[j], linewidth=2.5)

#     y_means2 = [np.mean([item['is_greater_than_threshold'] for item in data2[attack_time]]) for attack_time in attack_times]
#     y_stds2 = [np.std([item['is_greater_than_threshold'] for item in data2[attack_time]]) for attack_time in attack_times]
#     plt.errorbar(x, y_means2, yerr=y_stds2, label=watermark_type2, fmt='-o')  # Use errorbar here
                    
# # Create the scatter plot
# for watermark_type, data,watermark_type, data2 in zip(data_dict.items(), data_dict_2.items()):
    
#     # x = list(data.keys())
#     x = [attack_times_mapping[attack_time] for attack_time in data.keys()]  # Change this line
#     y = [sum(item['is_greater_than_threshold'] for item in data[attack_time]) / len(data[attack_time]) for attack_time in attack_times]
#     # plt.plot(x, y, label=watermark_type)
#     plt.errorbar(x, y_means, yerr=y_stds, label=watermark_type, fmt='-o')  # Use errorbar here
    # plt.scatter(x, y)  # Add this line
# Set the title and labels
# plt.title('TPR under Multi ChatGPT 3.5 Paraphase')

plt.xlabel('GPT Attack Times')
ax.set_ylabel('TPR (with FPR = 0.01)')

plt.xticks(range(1, 6))  # Add this line
ax.grid(axis='y', linestyle='--')  # Add this line

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)

# Set the y-axis range
plt.ylim([0, 0.31])  # Add this line
# Add a legend
plt.legend(loc='upper right',ncol=3)  # Change this line
plt.tight_layout()  # Add this line

# Show the plot
plt.savefig(f'./plot/newplot/output/multi_gpt_attack_2.pdf')
print(f'./plot/newplot/output/multi_gpt_attack_2.pdf')