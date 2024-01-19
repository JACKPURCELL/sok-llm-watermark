#!/bin/bash
#===============================================================================
# usecommand: nohup bash topp.sh > topp.log 2>&1 &
#===============================================================================
models=("opt" "llama")
watermark_types=("xuandong23b" "rohith23" "lean23" "kiyoon23" "xiaoniu23")

# "xuandong23b" "rohith23" "lean23" "kiyoon23" "xiaoniu23"

top_ps=("0.7")

for watermark_type in "${watermark_types[@]}"; do
    echo "start $watermark_type"
    for top_p in "${top_ps[@]}"; do

        echo "start $temp"
        CUDA_VISIBLE_DEVICES=2,0,1,3 python watermark_reliability_release/generation_pipeline.py --top_p "$top_p" --watermark "$watermark_type" --dataset_name c4 --run_name gen-c4-"$watermark_type"-opt-topp"$top_p" --model_name_or_path facebook/opt-1.3b --min_generations 1000 --max_new_tokens 100 --output_dir "$watermark_type"/c4/opt/gen-topp-"$top_p" 
        CUDA_VISIBLE_DEVICES=2,0,1,3 python watermark_reliability_release/generation_pipeline.py --top_p "$top_p" --watermark "$watermark_type" --dataset_name c4 --run_name gen-c4-"$watermark_type"-llama-topp"$top_p" --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 --max_new_tokens 100 --output_dir "$watermark_type"/c4/llama/gen-topp-"$top_p"
        echo "finish $temp"

    done
    echo "finish $watermark_type"

done
echo "finish all"
