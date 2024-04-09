#!/bin/bash
watermark_types=("john23" "xuandong23b" "rohith23" "lean23" "aiwei23" "aiwei23b" "scott22" "xiaoniu23")
for watermark_type in "${watermark_types[@]}"; do
    tar -czvf "${watermark_type}.tar.gz" "/home/ljc/sok-llm-watermark/runs/token_200/${watermark_type}/c4/opt/dipper_l40_o0/dipp_roberta_finetuned"
done

# Create a large tar package containing all the .tar.gz files
tar -cvf large_package.tar *.tar.gz



# tar -xzvf large_package.tar -C "/home/ljc/sok-llm-watermark/runs/token_200/"