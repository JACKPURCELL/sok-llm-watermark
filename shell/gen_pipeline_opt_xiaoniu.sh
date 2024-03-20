#!/bin/bash
#===============================================================================
# usecommand nohup bash shell/gen_pipeline_opt_xiaoniu.sh > shell/gen_pipeline_opt_xiaoniu23eva.log 2>&1 &
#===============================================================================

watermark_types=("xiaoniu23")

models=("opt")
gpus=("0")

for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"s
        
        # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpus" python命令
        # CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project xiaoniu_new --run_name gen-c4-"$watermark_type"-"$model"  --output_dir token_200/"$watermark_type"/c4/"$model" --min_generations 1000 --max_new_tokens 200 --use_sampling False --generation_batch_size 1 --num_beams 1 --model_name_or_path facebook/opt-1.3b 
        CUDA_VISIBLE_DEVICES="$gpus" nohup python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project xiaoniu_new --run_name eva-c4-"$watermark_type"-"$model"  --input_dir ~/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model" --overwrite_args True &
    echo "finish $model"

    done
echo "finish $watermark_type"
done