#!/bin/bash
#===============================================================================
# usecommand  nohup bash shell/0331_mix_lean23_eva_add.sh > shell/0331_mix_lean23_eva_add.sh.log 2>&1 &
#===============================================================================

# 定义一个包含不同watermark类型的数组
# SUPPORTED_ATTACK_METHODS = [ "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym"]
watermark_types=("lean23" ) 
# watermark_types=("john23"	"xuandong23b"	"rohith23"	"lean23"	"aiwei23"		"xiaoniu23")
#gpu1
# watermark_types=(	"xuandong23b"	"kiyoon23"	"xiaoniu23")
#gpu2
# watermark_types=(	"lean23" "aiwei23" "rohith23")

models=("opt")
# attack_types=( "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym")
attack_types=( "swap" "translation")
synonym_probs=( "0.4" )
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" "40")
cp_attack_insertion_lens=("10")
gpus=("1")
tokens=("token_200")
# cp_attack_types=("single-single" "triple-single")
# 遍历数组中的每个元素
# CUDA_VISIBLE_DEVICES=0,2,0  CUDA_VISIBLE_DEVICES=0,2 python watermark_reliability_release/evaluation_pipeline.py --wandb True --only_attack_zscore True   --overwrite_output_file True --watermark aiwei23 --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name eva-aiwei23-hc3-dipper --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/aiwei23/hc3/dipper --evaluation_metrics all --wandb_project eva-lean23-attack --overwrite_args True
for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        for token in  "${tokens[@]}"; do
    echo "start $token"


   

        # copypaste
        for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --overwrite_output_file True --watermark "$watermark_type" --run_name copypaste-1-"$cp_attack_insertion_len"-neweva-"$watermark_type"-hc3-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/hc3/"$model"/copypaste-1-"$cp_attack_insertion_len"  --wandb_project temporary --overwrite_args True
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --overwrite_output_file True --watermark "$watermark_type" --run_name copypaste-3-"$cp_attack_insertion_len"-neweva-"$watermark_type"-hc3-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/hc3/"$model"/copypaste-3-"$cp_attack_insertion_len" --wandb_project temporary --overwrite_args True
        done

    done
    done
    echo "finish $model"
done
echo "finish $watermark_type"


