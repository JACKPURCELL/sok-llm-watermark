#!/bin/bash
#===============================================================================
# usecommand nohup bash shell/gen_pipeline_opt_lean23_50.sh > shell/gen_pipeline_opt_lean23_50.log 2>&1 &
#===============================================================================

watermark_types=("lean23")

models=("opt")
gpus=("1")

for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        
        # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpus" python命令
        CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project temporary --run_name gen-c4-"$watermark_type"-"$model"  --output_dir token_50_new/"$watermark_type"/c4/"$model" --min_generations 700 --max_new_tokens 50 --generation_batch_size 1 --model_name_or_path facebook/opt-1.3b
        CUDA_VISIBLE_DEVICES="$gpus" nohup python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project temporary --run_name eva-c4-"$watermark_type"-"$model"  --input_dir /home/ljc/sok-llm-watermark/runs/token_50_new/"$watermark_type"/c4/"$model" --overwrite_args True &

    done
    echo "finish $model"
done
echo "finish $watermark_type"
