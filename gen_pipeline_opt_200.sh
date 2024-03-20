#!/bin/bash
#===============================================================================
# usecommand nohup bash gen_pipeline_opt_200.sh > gen_pipeline_opt_aiwei23b_100.log 2>&1 &
#===============================================================================

# watermark_types=("john23" "xiaoniu23" "lean23")
watermark_types=("aiwei23b")
#"john23" "lean23"
#"rohith23" "xiaoniu23" "xuandong23b" "kiyoon23" 
### Aiwei23
###"john23" "kiyoon23" "lean23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23"
models=("opt")
gpus=("3")

# cp_attack_types=("single-single" "triple-single")
# 遍历数组中的每个元素
# CUDA_VISIBLE_DEVICES="$gpus",0 nohup CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --wandb True --only_attack_zscore True --overwrite_output_file True --watermark aiwei23 --only_attack_zscore True --overwrite_output_file True --watermark "$watermark_type" --run_name eva-aiwei23-hc3-dipper --input_dir /home/ljc/sok-llm-watermark/runs/aiwei23/hc3/dipper --evaluation_metrics all &
for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"

            # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpus" python命令
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project gen-c4-v2.0-50tokens --run_name gen-c4-"$watermark_type"-"$model"  --output_dir "token_50/$watermark_type"/c4/"$model" --min_generations 700 --max_new_tokens 50 --generation_batch_size 1 --model_name_or_path facebook/opt-1.3b --generation_batch_size 1 --dataset_name c4
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project gen-c4-v2.0-50tokens --run_name eva-c4-"$watermark_type"-"$model"  --input_dir "/home/ljc/sok-llm-watermark/runs/token_50/$watermark_type"/c4/"$model"  --overwrite_args True  --overwrite_output_file True --lower_tolerance_T 25 --upper_tolerance_T 50 

    done
    
    echo "finish $model"
done
echo "finish $watermark_type"




