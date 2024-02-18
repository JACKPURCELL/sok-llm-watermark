import json
import math
import os
import glob

# 定义目录路径
directory = 'runs/token_50/lean23/c4/opt/'

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# 使用glob获取目录及其子目录下的所有 gen_table_w_metrics.jsonl 文件
file_list = glob.glob(os.path.join(directory, '**', 'gen_table_w_metrics.jsonl'), recursive=True)

def modify_file(file):
    temp_file = "temp.jsonl"
    with open(file, 'r') as read_obj, open(temp_file, 'w') as write_obj:
        for line in read_obj:
            data = json.loads(line)
            z_score = data["baseline_completion_confidence"]
            prediction = 1 if data["baseline_completion_prediction"] else -1
            product = sigmoid(z_score * prediction)
            data["baseline_completion_z_score"] = product
            # data["baseline_completion_confidence"] = z_score
            try:
                z_score = data["w_wm_output_confidence"]
                prediction = 1 if data["w_wm_output_prediction"] else -1
                product = sigmoid(z_score * prediction)
                data["w_wm_output_z_score"] = product
                # data["w_wm_output_confidence"] = z_score

            except:
                z_score = data["w_wm_output_attacked_confidence"]
                prediction = 1 if data["w_wm_output_attacked_prediction"] else -1
                product = sigmoid(z_score * prediction)
                data["w_wm_output_attacked_z_score"] = product
                # data["w_wm_output_attacked_confidence"] = z_score
            write_obj.write(json.dumps(data) + "\n")
    os.remove(file)
    os.rename(temp_file, file)

for file in file_list:
    modify_file(file)

print(file_list)