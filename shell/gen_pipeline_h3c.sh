# !/bin/bash
# ===============================================================================
# usecommand nohup bash shell/gen_pipeline_h3c.sh > shell/gen_pipeline_h3c_rohith23.log 2>&1 &
# ===============================================================================
echo "Current date and time: $(date)"

watermark_types=("rohith23" )
# watermark_types=("john23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23" "aiwei23b")

models=("opt")
gpus=("0")

for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        
        # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpus" python命令
        # CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project hc3 --dataset_name hc3 --run_name gen-hc3-"$watermark_type"-"$model"  --output_dir token_200/"$watermark_type"/hc3/"$model" --min_generations 700 --max_new_tokens 200 --use_sampling False --generation_batch_size 1 --num_beams 1 --model_name_or_path facebook/opt-1.3b 
        CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project hc3 --run_name eva-hc3-"$watermark_type"-"$model"  --input_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model" --overwrite_args True 
    echo "finish $model"

    done
echo "finish $watermark_type"
done
echo "Current date and time: $(date)"

watermark_types=("rohith23" )
# watermark_types=("john23" "rohith23"  "xuandong23b" "aiwei23" "aiwei23b")

models=("llama")
gpus=("0")

for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"s
        
        # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpus" python命令
        # CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project hc3 --dataset_name hc3 --run_name gen-hc3-"$watermark_type"-"$model"  --output_dir token_200/"$watermark_type"/hc3/"$model" --min_generations 700 --max_new_tokens 200 --use_sampling False --generation_batch_size 1 --num_beams 1 --model_name_or_path meta-llama/Llama-2-7b-chat-hf
        CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project hc3 --run_name eva-hc3-"$watermark_type"-"$model"  --input_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model" --overwrite_args True 
    echo "finish $model"

    done
echo "finish $watermark_type"
done


# watermark_types=("xiaoniu23" )

# models=("llama")
# gpus=("0,1,2")

# for watermark_type in "${watermark_types[@]}"; do
# echo "start $watermark_type"
#     for model in "${models[@]}"; do
#     echo "start $model"s
        
#         # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpus" python命令
#         CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project c4 --dataset_name c4 --run_name gen-c4-"$watermark_type"-"$model"  --output_dir token_200/"$watermark_type"/c4/"$model" --min_generations 700 --max_new_tokens 200 --use_sampling False --generation_batch_size 1 --num_beams 1 --model_name_or_path meta-llama/Llama-2-7b-chat-hf
#         CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project c4 --run_name eva-c4-"$watermark_type"-"$model"  --input_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model" --overwrite_args True 
#     echo "finish $model"

#     done
# echo "finish $watermark_type"
# done

# echo "Current date and time: $(date)"