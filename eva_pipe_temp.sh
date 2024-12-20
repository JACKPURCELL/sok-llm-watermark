#!/bin/bash
#===============================================================================
# usecommand nohup bash eva_pipe_temp.sh > eva_pipe.log 2>&1 &
#===============================================================================
models=("opt" "llama")
watermark_types=("xuandong23b")

CUDA_VISIBLE_DEVICES=2,0,1 python watermark_reliability_release/generation_pipeline.py --sampling_temp 0.4 --watermark xuandong23b --dataset_name c4 --run_name gen-c4-xuandong23b-llama-temp0.4 --model_name_or_path meta-llama/Llama-2-7b-chat-hf --min_generations 1000 --max_new_tokens 100 --output_dir xuandong23b/c4/llama/gen-0.4

temps=("0.4" "1.0" "1.3")
for watermark_type in "${watermark_types[@]}"; do
    for temp in "${temps[@]}"; do

    CUDA_VISIBLE_DEVICES=2,0,1 python watermark_reliability_release/evaluation_pipeline.py  --watermark "$watermark_type" --dataset_name c4 --run_name eva-c4-"$watermark_type"-llama-temp"$temp" --input_dir "$watermark_type/c4/llama/gen-$temp"
    CUDA_VISIBLE_DEVICES=2,0,1 python watermark_reliability_release/evaluation_pipeline.py  --watermark "$watermark_type" --dataset_name c4 --run_name eva-c4-"$watermark_type"-opt-temp"$temp" --input_dir "$watermark_type/c4/opt/gen-$temp" 

    done
done


# 定义一个包含不同watermark类型的数组
# SUPPORTED_ATTACK_METHODS = [ "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym"]
watermark_type="john23"
models=("opt" "llama")
# attack_types=( "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym")
attack_types=( "swap")
synonym_probs=("0.1" "0.2" "0.4" "0.8" "1.0")
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" "40" "60")
cp_attack_insertion_lens=("10" "25")
# cp_attack_types=("single-single" "triple-single")
# 遍历数组中的每个元素
# CUDA_VISIBLE_DEVICES=1,0 nohup python watermark_reliability_release/evaluation_pipeline.py --wandb True --only_attack_zscore True --watermark aiwei23 --only_attack_zscore True --watermark "$watermark_type" --run_name eva-aiwei23-hc3-dipper --input_dir /home/jkl6486/sok-llm-watermark/runs/aiwei23/hc3/dipper --evaluation_metrics all &

for model in "${models[@]}"; do
  

    for attack_type in "${attack_types[@]}"; do
        # 对每个元素执行python命令
        # python main.py --only_attack_zscore True --watermark "$attack_type"
        python watermark_reliability_release/evaluation_pipeline.py --watermark "$watermark_type" --run_name "$attack_type"-eva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/"$attack_type"
    done


    #synonym
    for synonym_prob in "${synonym_probs[@]}"; do
        python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --watermark "$watermark_type" --run_name synonym-"$synonym_prob"-eva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/synonym-"$synonym_prob"
    done

    #copypaste
    for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
        python watermark_reliability_release/evaluation_pipeline.py --cp_attack_types single-single --only_attack_zscore True --watermark "$watermark_type" --run_name copypaste-1-"$cp_attack_insertion_len"-eva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/copypaste-1-"$cp_attack_insertion_len" 
        python watermark_reliability_release/evaluation_pipeline.py --cp_attack_types triple-single --only_attack_zscore True --watermark "$watermark_type" --run_name copypaste-3-"$cp_attack_insertion_len"-eva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/copypaste-3-"$cp_attack_insertion_len"
    done

    #helm
    for helm_attack_method in "${helm_attack_methods[@]}"; do
        python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --watermark "$watermark_type" --run_name helm-"$helm_attack_method"-eva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/"$attack_type" 
    done

    #dipper need 48gb memory
    # for dipper_lex in "${dipper_lexs[@]}"; do
    #     python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --watermark "$watermark_type" --run_name dipper-eva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0 
    # done
    # python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --watermark "$watermark_type" --run_name dipper-eva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/dipper_l60_o20 

done