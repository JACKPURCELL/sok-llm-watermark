import json
import os
import glob

# 定义目录路径
directory = 'runs/token_100/rohith23/c4/opt/'

# 使用glob获取目录及其子目录下的所有 gen_table_w_metrics.jsonl 文件
file_list = glob.glob(os.path.join(directory, '**', 'gen_table_w_metrics.jsonl'), recursive=True)

def modify_file(file):
    temp_file = "temp.jsonl"
    with open(file, 'r') as read_obj, open(temp_file, 'w') as write_obj:
        for line in read_obj:
            data = json.loads(line)
            try:
                p_value = data['w_wm_output_p-value']
                data["w_wm_output_prediction"] = p_value < 0.02
            except:
                p_value = data['w_wm_output_attacked_p-value']
                data["w_wm_output_attacked_prediction"] = p_value < 0.02
            write_obj.write(json.dumps(data) + "\n")
    os.remove(file)
    os.rename(temp_file, file)

for file in file_list:
    modify_file(file)

print(file_list)