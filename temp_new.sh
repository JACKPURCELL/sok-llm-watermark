#!/bin/bash
#===============================================================================
# usecommand: nohup bash temp.sh > temp.log 2>&1 &
#===============================================================================
models=("opt" "llama")
watermark_types=("john23")

# "xuandong23b" "rohith23" "lean23" "kiyoon23" "xiaoniu23"

temps=("1.0" "1.3")

for watermark_type in "${watermark_types[@]}"; do
    echo "start $watermark_type"
    for temp in "${temps[@]}"; do

        echo "start $temp"
        CUDA_VISIBLE_DEVICES=2,0,1,3 python watermark_reliability_release/generation_pipeline.py --sampling_temp "$temp" --watermark "$watermark_type" --dataset_name c4 --run_name gen-c4-"$watermark_type"-opt-temp"$temp" --model_name_or_path facebook/opt-1.3b --min_generations 1000 --max_new_tokens 100 --output_dir "$watermark_type"/c4/opt/gen-"$temp" 
        CUDA_VISIBLE_DEVICES=2,0,1,3 python watermark_reliability_release/generation_pipeline.py --sampling_temp "$temp" --watermark "$watermark_type" --dataset_name c4 --run_name gen-c4-"$watermark_type"-llama-temp"$temp" --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 --max_new_tokens 100 --output_dir "$watermark_type"/c4/llama/gen-"$temp"
        echo "finish $temp"

    done
    echo "finish $watermark_type"

done
echo "finish all"
