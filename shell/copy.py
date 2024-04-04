import shutil

# 水印类型列表
watermark_types = ["john23", "rohith23", "xiaoniu23", "xuandong23b", "aiwei23", "aiwei23b", "scott22", "lean23"]

# 遍历每个水印类型
for watermark_type in watermark_types:
    # 源文件路径
    src = f"/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/c4/opt/gen_table_meta.json"

    # 目标文件路径
    dst = f"/home/jkl6486/sok-llm-watermark/runs/token_200/{watermark_type}/c4/opt/truncated/gen_table_meta.json"

    # 复制文件
    shutil.copy(src, dst)