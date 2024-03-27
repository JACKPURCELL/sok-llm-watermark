#!/bin/bash
#===============================================================================
# usecommand nohup bash gen_pipeline_opt.sh > gen_pipeline_opt_aiwei23b_temp.log 2>&1 &
#===============================================================================

watermark_types=("john23")
#"john23" "lean23"
#"rohith23" "xiaoniu23" "xuandong23b" "kiyoon23" 
### Aiwei23
###"john23" "kiyoon23" "lean23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23"
models=("llama")
gpus=("0, 1, 2, 3")
temps=("1.3" "0.7")

# cp_attack_types=("single-single" "triple-single")
# 遍历数组中的每个元素
# CUDA_VISIBLE_DEVICES="$gpus",0 nohup CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --wandb True --only_attack_zscore True --overwrite_output_file True --watermark aiwei23 --only_attack_zscore True --overwrite_output_file True --watermark "$watermark_type" --run_name eva-aiwei23-hc3-dipper --input_dir /home/jkl6486/sok-llm-watermark/runs/aiwei23/hc3/dipper --evaluation_metrics all &
for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        for temp in "${temps[@]}"; do
        echo "start $temp"
            # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpus" python命令
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project gen-c4-v2.0-200tokens-"$temp"temp --run_name gen-c4-"$watermark_type"-"$model"  --output_dir "token_200/$watermark_type"/c4/"$model"-"$temp" --min_generations 700 --max_new_tokens 200 --generation_batch_size 1 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --use_sampling True --sampling_temp "$temp"
            CUDA_VISIBLE_DEVICES="$gpus" nohup python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project gen-c4-v2.0-200tokens-"$temp"temp --run_name eva-c4-"$watermark_type"-"$model"  --input_dir "/home/jkl6486/sok-llm-watermark/runs/token_200/$watermark_type"/c4/"$model"-"$temp"  --overwrite_args True  --overwrite_output_file True   --lower_tolerance_T 50 --upper_tolerance_T 100 &
        done
    done
    echo "finish $model"
done
echo "finish $watermark_type"
