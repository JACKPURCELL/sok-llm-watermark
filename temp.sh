#!/bin/bash
#===============================================================================
# usecommand: nohup bash temp.sh > temp_kiyoon23.log 2>&1 &
#===============================================================================
models=("opt" "llama")
watermark_types=("kiyoon23")

# CUDA_VISIBLE_DEVICES=2,0,1 python watermark_reliability_release/generation_pipeline.py --sampling_temp 0.4 --watermark xuandong23b --dataset_name c4 --run_name gen-c4-xuandong23b-llama-temp0.4 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 --max_new_tokens 100 --output_dir xuandong23b/c4/llama/gen-0.4

temps=("0.4" "1.0" "1.3")
for watermark_type in "${watermark_types[@]}"; do
    for temp in "${temps[@]}"; do

    CUDA_VISIBLE_DEVICES=2,0,1,3 python watermark_reliability_release/generation_pipeline.py --sampling_temp "$temp" --watermark "$watermark_type" --dataset_name c4 --run_name gen-c4-"$watermark_type"-llama-temp"$temp" --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 --max_new_tokens 100 --output_dir "$watermark_type/c4/llama/gen-$temp"
    # CUDA_VISIBLE_DEVICES=2,0,1,3 python watermark_reliability_release/generation_pipeline.py --sampling_temp "$temp" --watermark "$watermark_type" --dataset_name c4 --run_name gen-c4-"$watermark_type"-opt-temp"$temp" --model_name_or_path facebook/opt-1.3b --min_generations 1000 --max_new_tokens 100 --output_dir "$watermark_type/c4/opt/gen-$temp" 

    done
done
