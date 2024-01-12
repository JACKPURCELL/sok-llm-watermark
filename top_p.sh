#!/bin/bash
#===============================================================================
# usecommand: nohup bash top_p.sh > top_p.log 2>&1 &
#===============================================================================
models=("opt" "llama")
watermark_types=("john23"	"xuandong23b"	"rohith23"	"lean23"	"aiwei23"	"kiyoon23"	"xiaoniu23")


top_p=("0.7")
for watermark_type in "${watermark_types[@]}"; do
    for top_p in "${top_ps[@]}"; do

    CUDA_VISIBLE_DEVICES=2,0,1,3 python watermark_reliability_release/generation_pipeline.py --watermark "$watermark_type" --dataset_name c4 --run_name gen-c4-"$watermark_type"-llama-top_p"$top_p" --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 --max_new_tokens 100 --top_p "$top_p" --output_dir "$watermark_type/c4/llama/gen-top_p-$top_p"
    CUDA_VISIBLE_DEVICES=2,0,1,3 python watermark_reliability_release/generation_pipeline.py --watermark "$watermark_type" --dataset_name c4 --run_name gen-c4-"$watermark_type"-opt-top_p"$top_p" --model_name_or_path facebook/opt-1.3b --min_generations 1000 --max_new_tokens 100 --top_p "$top_p" --output_dir "$watermark_type/c4/opt/gen-top_p-$top_p" 

    done
done