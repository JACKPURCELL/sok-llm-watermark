#!/bin/bash
#===============================================================================
# usecommand nohup bash gen_pipeline_opt.sh > gen_pipeline_opt_v2.log 2>&1 &
#===============================================================================

watermark_types=("lean23"	"xiaoniu23")

models=("opt" )
gpus=("1,2,3")

# cp_attack_types=("single-single" "triple-single")
# 遍历数组中的每个元素
# CUDA_VISIBLE_DEVICES="$gpus",0 nohup CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --wandb True --only_attack_zscore True --overwrite_output_file True --watermark aiwei23 --only_attack_zscore True --overwrite_output_file True --watermark "$watermark_type" --run_name eva-aiwei23-hc3-dipper --input_dir /home/jkl6486/sok-llm-watermark/runs/aiwei23/hc3/dipper --evaluation_metrics all &
for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        
        # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpus" python命令
        CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project gen-c4-v2.0-1000 --run_name gen-c4-"$watermark_type"-"$model"  --output_dir "$watermark_type"/c4/"$model" --min_generations 1000 --max_new_tokens 200 --use_sampling False --generation_batch_size 1 --num_beams 1 --model_name_or_path facebook/opt-1.3b
        CUDA_VISIBLE_DEVICES="$gpus" nohup python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project gen-c4-v2.0-1000 --run_name eva-c4-"$watermark_type"-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"  --overwrite_args True &


    done
    echo "finish $model"
done
echo "finish $watermark_type"
