#!/bin/bash
#===============================================================================
# usecommand nohup bash gen_pipeline_llama.sh > gen_pipeline_llama_v4.log 2>&1 &
#===============================================================================

watermark_types=("xiaoniu23" )

models=("llama")
gpus=("0,1,2,3")
for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        
 
        CUDA_VISIBLE_DEVICES="$gpus" nohup python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project gen-c4-v2.0-500-llama --run_name eva-c4-"$watermark_type"-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/token_200/"$watermark_type"/c4/"$model" --overwrite_args True &

    done
    echo "finish $model"
done
echo "finish $watermark_type"
