#!/bin/bash
#===============================================================================
# usecommand nohup bash shell/gen_pipeline_opt_aiwei23.sh > shell/gen_pipeline_opt_aiwei23.log 2>&1 &
#===============================================================================

watermark_types=("aiwei23")

models=("opt")
gpus=("0,1,2,3")

for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        
        # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpus" python命令
        # CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project gen-c4-v2.0-500-llama --run_name gen-c4-"$watermark_type"-"$model"  --output_dir token_200/"$watermark_type"/c4/"$model" --min_generations 700 --max_new_tokens 200 --use_sampling False --generation_batch_size 1 --num_beams 1 --model_name_or_path meta-llama/Llama-2-7b-chat-hf 
        CUDA_VISIBLE_DEVICES="$gpus" nohup python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project temporary --run_name eva-c4-"$watermark_type"-"$model"  --input_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model" --overwrite_args True &

    done
    echo "finish $model"
done
echo "finish $watermark_type"
