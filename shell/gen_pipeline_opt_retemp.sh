#!/bin/bash
#===============================================================================
# usecommand nohup bash shell/gen_pipeline_opt_retemp.sh > shell/gen_pipeline_opt_retemp_xiaoniu.log 2>&1 &
#===============================================================================

watermark_types=("xiaoniu23")
#"john23" "lean23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23" "aiwei23b"

models=("opt")
gpus=("1, 2, 3")
temps=("1.0")

for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        for temp in "${temps[@]}"; do
        echo "start $temp"
        
            # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpus" python命令
            #CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project regen-temp --run_name gen-c4-"$watermark_type"-"$model"-temp"$temp"  --output_dir token_200/"$watermark_type"/c4/"$model"-temp"$temp" --min_generations 700 --max_new_tokens 200 --use_sampling True --generation_batch_size 1 --model_name_or_path facebook/opt-1.3b --sampling_temp "$temp" --top_p 0.9 
            CUDA_VISIBLE_DEVICES="" nohup python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project reeva-temp --run_name eva-c4-"$watermark_type"-"$model"-temp"$temp"  --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"-temp"$temp" --overwrite_args True &
        
        done
    done
    echo "finish $model"
done
echo "finish $watermark_type"
