#!/bin/bash
#===============================================================================
# usecommand: nohup bash token.sh > token.log 2>&1 &
#===============================================================================
models=("opt" "llama")
watermark_types=("john23")


tokens=("25" "50" "100" "200")
for watermark_type in "${watermark_types[@]}"; do
    for token in "${tokens[@]}"; do

    CUDA_VISIBLE_DEVICES=2,0,1 python watermark_reliability_release/generation_pipeline.py --watermark "$watermark_type" --dataset_name c4 --run_name gen-c4-"$watermark_type"-llama-token"$token" --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 --max_new_tokens "$token" --output_dir "$watermark_type/c4/llama/gen-token-$token"
    CUDA_VISIBLE_DEVICES=2,0,1 python watermark_reliability_release/generation_pipeline.py --watermark "$watermark_type" --dataset_name c4 --run_name gen-c4-"$watermark_type"-opt-token"$token" --model_name_or_path facebook/opt-1.3b --min_generations 1000 --max_new_tokens "$token" --output_dir "$watermark_type/c4/opt/gen-token-$token" 

    done
done