#!/bin/bash
#===============================================================================
# usecommand nohup bash eva_pipe_z.sh > eva_pipe_xiaoniu23n.log 2>&1 &
#===============================================================================

# 定义一个包含不同watermark类型的数组
# SUPPORTED_ATTACK_METHODS = [ "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym"]
watermark_types=("xiaoniu23") 
# watermark_types=("john23"	"xuandong23b"	"rohith23"	"lean23"	"aiwei23"	"kiyoon23"	"xiaoniu23")
#gpu1
# watermark_types=(	"xuandong23b"	"kiyoon23"	"xiaoniu23")
#gpu2
# watermark_types=(	"lean23" "aiwei23" "rohith23")

models=("opt" "llama")
# attack_types=( "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym")
attack_types=( "swap")
synonym_probs=("0.1" "0.2" "0.4" "0.8" "1.0")
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" "40" "60")
cp_attack_insertion_lens=("10" "25")
# cp_attack_types=("single-single" "triple-single")
# 遍历数组中的每个元素
# CUDA_VISIBLE_DEVICES=0,1,2,0 nohup CUDA_VISIBLE_DEVICES=0,1,2 python watermark_reliability_release/evaluation_pipeline.py --wandb True --overwrite_output_file True --only_attack_zscore True  --watermark aiwei23 --overwrite_output_file True --only_attack_zscore True  --watermark "$watermark_type" --run_name eva-aiwei23-hc3-dipper --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/aiwei23/hc3/dipper --evaluation_metrics all &
for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        # swap
        for attack_type in "${attack_types[@]}"; do
            # 对每个元素执行CUDA_VISIBLE_DEVICES=0,1,2 python命令
            # CUDA_VISIBLE_DEVICES=0,1,2 python main.py --overwrite_output_file True --only_attack_zscore True  --watermark "$attack_type"
            CUDA_VISIBLE_DEVICES=0,1,2 python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --only_attack_zscore True  --watermark "$watermark_type" --run_name "$attack_type"-evaz-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$watermark_type"/c4/"$model"/"$attack_type"
        done


        #synonym
        for synonym_prob in "${synonym_probs[@]}"; do
            CUDA_VISIBLE_DEVICES=0,1,2 python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --only_attack_zscore True  --watermark "$watermark_type" --run_name synonym-"$synonym_prob"-evaz-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$watermark_type"/c4/"$model"/synonym-"$synonym_prob"
        done

        #copypaste
        for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
            CUDA_VISIBLE_DEVICES=0,1,2 python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --only_attack_zscore True  --watermark "$watermark_type" --run_name copypaste-1-"$cp_attack_insertion_len"-evaz-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$watermark_type"/c4/"$model"/copypaste-1-"$cp_attack_insertion_len" 
            CUDA_VISIBLE_DEVICES=0,1,2 python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --only_attack_zscore True  --watermark "$watermark_type" --run_name copypaste-3-"$cp_attack_insertion_len"-evaz-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$watermark_type"/c4/"$model"/copypaste-3-"$cp_attack_insertion_len"
        done

        #helm
        for helm_attack_method in "${helm_attack_methods[@]}"; do
            CUDA_VISIBLE_DEVICES=0,1,2 python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --only_attack_zscore True  --watermark "$watermark_type" --run_name helm-"$helm_attack_method"-evaz-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$watermark_type"/c4/"$model"/"$helm_attack_method" 
        done

        #dipper need 48gb memory
        for dipper_lex in "${dipper_lexs[@]}"; do
            CUDA_VISIBLE_DEVICES=0,1,2 python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --only_attack_zscore True  --watermark "$watermark_type" --run_name dipper-evaz-"$watermark_type"-c4-"$model"-dipper_l"$dipper_lex"_o0   --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0 
        done
        CUDA_VISIBLE_DEVICES=0,1,2 python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --only_attack_zscore True  --watermark "$watermark_type" --run_name dipper-evaz-"$watermark_type"-c4-"$model"-dipper_l60_o20   --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$watermark_type"/c4/"$model"/dipper_l60_o20 

    done
    echo "finish $model"
done
echo "finish $watermark_type"
