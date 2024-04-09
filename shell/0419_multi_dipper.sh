#!/bin/bash
#===============================================================================
# usecommand nohup bash shell/0419_multi_dipper.sh > shell/0419_multi_dipper_scott22.log 2>&1 &
#===============================================================================

# SUPPORTED_ATTACK_METHODS = [ "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym"]
watermark_types=("scott22")
# john23 unrun
# "xiaoniu23" "xuandong23b" "aiwei23" "aiwei23b"
#"john23" "lean23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23" "kiyoon23" "|"
models=("opt")
# attack_types=( "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym")
# ===============================================================================
dipper_lexs=("40")
gpus=("2")
tokens=( "token_200" )
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
            for rep_num in {2..5}; do
                mkdir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/"dipper_${dipper_lexs}_rep${rep_num}"

                cp /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/gen_table_meta.json /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/"dipper_${dipper_lexs}_rep${rep_num}"/
                if [ $rep_num -eq 2 ]
                then
                    python /home/jkl6486/sok-llm-watermark/watermark_reliability_release/multiattack_preprocessing.py --input_path /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/"dipper_l${dipper_lexs}_o0"/gen_table_attacked.jsonl --output_path /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/"dipper_${dipper_lexs}_rep${rep_num}"/gen_table.jsonl
                else
                    python /home/jkl6486/sok-llm-watermark/watermark_reliability_release/multiattack_preprocessing.py --input_path /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/"dipper_${dipper_lexs}_rep$((rep_num - 1))"/gen_table_attacked.jsonl --output_path /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/"dipper_${dipper_lexs}_rep${rep_num}"/gen_table.jsonl
                fi
                # dipper need 48gb memory
                for dipper_lex in "${dipper_lexs[@]}"; do
                    CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/"dipper_${dipper_lexs}_rep${rep_num}"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/"dipper_${dipper_lexs}_rep${rep_num}" --attack_method dipper --lex "$dipper_lex" 
                done
            done    
        done
    done
    done
    echo "finish $watermark_type"
done
echo "finish all"
