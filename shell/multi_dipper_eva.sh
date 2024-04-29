#!/bin/bash
#===============================================================================
# usecommand nohup bash shell/multi_dipper_eva.sh > shell/0422_multidipper_eva1.log 2>&1 &
#===============================================================================

# SUPPORTED_ATTACK_METHODS = [ "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym"]
watermark_types=("john23" "rohith23" "xuandong23b" "scott22" "aiwei23b"  "xiaoniu23"  "aiwei23")
# watermark_types=("john23" "rohith23" "xuandong23b" "scott22")
# john23 unrun
# "xiaoniu23" "xuandong23b" "aiwei23" "aiwei23b"
#"john23" "lean23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23" "kiyoon23" "|"
models=("opt")
# attack_types=( "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym")
# ===============================================================================
dipper_lexs=("20")
gpus=("1")
tokens=( "token_200" )
attack_1="dipper_l40_o0"
# ===============================================================================
# attack_types=( "swap")
# synonym_probs=("0.1" "0.2" "0.4" "0.8" "1.0")
# helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
# dipper_lexs=("20" "40" "60")
# cp_attack_insertion_lens=("10" "25")
# ===============================================================================
# cp_attack_types=("single-single" "triple-single")
# 遍历数组中的每个元素
for watermark_type in "${watermark_types[@]}"; do
    echo "start $watermark_type"
    for model in "${models[@]}"; do
        for token in  "${tokens[@]}"; do
            # for rep_num in {2..5}; do
                cp /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/dipper_l40_o0/gen_table_attacked_meta.json /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/adv_perturb

                CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva --wandb False --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/adv_perturb --overwrite_args True 
            # done    
        done
    done
    echo "finish $watermark_type"
done
echo "finish all"



        
# runs/token_200/john23/c4/opt/adv_perturb

