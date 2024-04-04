#!/bin/bash
#===============================================================================
# usecommand nohup bash shell/0331_john_truncated.sh > shell/0331_john_truncated.log 2>&1 &
#===============================================================================

watermark_types=("john23")

models=("opt")
gpus=("3")

for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"s
        

        CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project truncated --run_name eva-c4-"$watermark_type"-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/truncated --overwrite_args True --evaluation_metrics z-score
    echo "finish $model"

    done
echo "finish $watermark_type"
done

