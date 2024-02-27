import matplotlib.pyplot as plt
import numpy as np
import math
import json
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer
plt.rcParams['font.size'] = 14  # 设置全局字体大小为14
watermark_types = ["john23","xuandong23b","aiwei23","lean23","rohith23","aiwei23b"]


tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
fig, ax = plt.subplots(figsize=(7, 7))

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # 添加这一行来定义颜色列表
data_dict = {}

for i, watermark_type in enumerate(watermark_types):
    file_path = f'/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/c4/opt/gen_table_w_metrics.jsonl'

    

    data_list = []
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
    # 获取预测的概率
            
    for data in data_list:
        for key in data.keys():
            if 'no_wm_output_vs_w_wm_output' in key:
                if key not in data_dict:
                    data_dict[key] = []
                data_dict[key].append(data[key])

    # 计算每个键的平均值和方差
    avg_var_dict = {key: (np.mean(values), np.var(values)) for key, values in data_dict.items()}

    # 画柱状图
    labels = avg_var_dict.keys()
    means = [avg for avg, var in avg_var_dict.values()]
    variances = [var for avg, var in avg_var_dict.values()]

    x = np.arange(len(labels))  # 标签的位置
    width = 0.35  # 柱子的宽度

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, means, width, label='Mean')
    rects2 = ax.bar(x + width/2, variances, width, label='Variance')

    # 添加一些文本标签，例如标题和轴标签
    ax.set_ylabel('Values')
    ax.set_title('Mean and Variance of each key')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()
    