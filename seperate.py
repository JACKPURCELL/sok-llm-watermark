import os
import json

# 打开并读取 JSON 文件
with open('~/sok-llm-watermark/runs/rohith23/c4/t300/dipper/gen_table_attacked.jsonl', 'r') as f:
    lines = f.readlines()

# 将文件内容分成每 5 行一个的小块
chunks = [lines[i:i + 5] for i in range(0, len(lines), 5)]

for i, chunk in enumerate(chunks):
    # 创建一个新的文件夹（如果尚不存在）
    if not os.path.exists(f'~/sok-llm-watermark/runs/rohith23/c4/t300/dipper/folder_{i}'):
        os.makedirs(f'~/sok-llm-watermark/runs/rohith23/c4/t300/dipper/folder_{i}')

    # 将块写入新的 JSON 文件中
    with open(f'~/sok-llm-watermark/runs/rohith23/c4/t300/dipper/folder_{i}/gen_table_attacked.jsonl', 'w') as f:
        for line in chunk:
            json.dump(json.loads(line), f)
            f.write('\n')
            
import os
import subprocess
import shutil

source_file = '~/sok-llm-watermark/runs/rohith23/c4/t300/dipper/gen_table_attacked_meta.json'


# 遍历每个文件夹
for i in range(len(chunks)):
    folder_name = f'~/sok-llm-watermark/runs/rohith23/c4/t300/dipper/folder_{i}'

    destination_file = os.path.join(folder_name, 'gen_table_attacked_meta.json')

    # 复制文件
    shutil.copy(source_file, destination_file)
    # 运行 Python 程序，文件作为 --input_dir 的参数输入
    command = f'CUDA_VISIBLE_DEVICES=0 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --watermark rohith23 --run_name eva-rohith23-c4-t300-dipper --input_dir {folder_name}  &'
    subprocess.run(command, shell=True)
    