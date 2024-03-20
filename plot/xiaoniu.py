import json
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm

# Initialize empty lists to store z_scores and true labels for all entries
z_scores_all = []
true_labels_all = []

# Read data from the JSON file
data_list = []
# watermark_types = ["john23","xuandong23b","aiwei23","xiaoniu23","rohith23","xiaoniu23"]
watermark_types = ["xiaoniu23"]


plt.figure()

for watermark_type in watermark_types:
    print(f"Processing {watermark_type}...")
    file = f'/home/ljc/sok-llm-watermark/runs/{watermark_type}/c4/opt/gen_table_w_metrics.jsonl'
    data_list = []
    with open(file, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))
    data_list = data_list[:500]
    # 获取预测的概率
    try:
        baseline_completion_z_score = [data["baseline_completion_best_sum_score"]  for data in data_list]
        baseline_completion_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in baseline_completion_z_score]
        w_wm_output_z_score = [data["w_wm_output_best_sum_score"] for data in data_list]
        w_wm_output_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in w_wm_output_z_score]
    except:
        continue
    # 创建一个新的图像
    plt.figure()

    # 画出 baseline_completion_z_score 的分布
    plt.hist(baseline_completion_z_score, bins=30, alpha=0.5, label='baseline_completion_z_score')

    # 画出 w_wm_output_z_score 的分布
    plt.hist(w_wm_output_z_score, bins=30, alpha=0.5, label='w_wm_output_z_score')

    # 添加图例
    plt.legend(loc='upper right')
    plt.savefig(f'./plot/xiaoniu_confidence.pdf')
    
 
 
 
 
# import json
# import math
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import interpolate
# from sklearn.metrics import roc_curve, auc
# from scipy.stats import norm

# # Initialize empty lists to store z_scores and true labels for all entries
# z_scores_all = []
# true_labels_all = []

# # Read data from the JSON file
# data_list = []
# # watermark_types = ["john23","xuandong23b","aiwei23","xiaoniu23","rohith23","xiaoniu23"]
# watermark_types = ["xiaoniu23"]


# plt.figure()

# for watermark_type in watermark_types:
#     print(f"Processing {watermark_type}...")
#     file = f'/home/ljc/sok-llm-watermark/runs/{watermark_type}/c4/opt/gen_table_w_metrics.jsonl'
#     data_list = []
#     with open(file, 'r') as f:
#         for line in f:
#             data_list.append(json.loads(line))
#     data_list = data_list[:500]
#     # 获取预测的概率
#     try:
#         baseline_completion_z_score_o = [data["baseline_completion_best_sum_score"] for data in data_list]
#         baseline_completion_prediction = [1 if data["baseline_completion_prediction"] else 0 for data in data_list]
#         baseline_completion_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in baseline_completion_z_score_o]
#         baseline_completion_z_score = [0.99999999 if score==1.0 else score for score in baseline_completion_z_score]
#         baseline_completion_z_score = [0.00000001 if score==0.0 else score for score in baseline_completion_z_score]
#         baseline_completion_z_score = [norm.ppf(score) if not math.isinf(score) and not math.isnan(score) else 0 for score in baseline_completion_z_score]
#         w_wm_output_z_score_o = [data["w_wm_output_best_sum_score"] for data in data_list]
#         w_wm_output_prediction = [1 if data["w_wm_output_prediction"] else 0 for data in data_list]
#         w_wm_output_z_score = [score if not math.isinf(score) and not math.isnan(score) else 0 for score in w_wm_output_z_score_o]
#         w_wm_output_z_score = [0.99999999 if score==1.0 else score for score in w_wm_output_z_score]
#         w_wm_output_z_score = [0.00000001 if score==0.0 else score for score in w_wm_output_z_score]
#         w_wm_output_z_score = [norm.ppf(score) if not math.isinf(score) and not math.isnan(score) else 0 for score in w_wm_output_z_score]
#     except:
#         continue

    
#     y_true = [0] * len(baseline_completion_z_score) + [1] * len(w_wm_output_z_score)
#     y_scores = baseline_completion_z_score + w_wm_output_z_score

#     # 计算 ROC 曲线
#     fpr, tpr, _ = roc_curve(y_true, y_scores)

#     # 计算 AUC
#     roc_auc = auc(fpr, tpr)

#     # 画出 ROC 曲线
#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic')
#     plt.legend(loc="lower right")       
#     plt.savefig(f'./plot/xiaoniu_auc.pdf')
    
#     # # 创建一个新的图像
#     # plt.figure()

#     # # 画出 baseline_completion_z_score 的分布
#     # plt.hist(baseline_completion_z_score, bins=30, alpha=0.5, label='baseline_completion_z_score')

#     # # 画出 w_wm_output_z_score 的分布
#     # plt.hist(w_wm_output_z_score, bins=30, alpha=0.5, label='w_wm_output_z_score')

#     # # 添加图例
#     # plt.legend(loc='upper right')
#     # plt.savefig(f'./plot/xiaoniu.pdf')