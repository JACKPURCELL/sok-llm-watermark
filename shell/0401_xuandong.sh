#!/bin/bash
#===============================================================================
# usecommand nohup bash shell/0401_xuandong.sh > shell/0401_xuandong.log 2>&1 &
#===============================================================================

# SUPPORTED_ATTACK_METHODS = [ "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym"]
# john23 unrun
# "xiaoniu23" "xuandong23b" "aiwei23" "aiwei23b"
#"john23" "lean23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23" "kiyoon23" "|"
models=("llama")
# num_beams=( "4" "8")

# attack_types=( "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym")
# ===============================================================================
attack_types=( "swap" "translation")
synonym_probs=( "0.4" )
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" "40")
cp_attack_insertion_lens=("10" "25")
gpus=("0")
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

watermark_types=( "xuandong23b")


for watermark_type in "${watermark_types[@]}"; do
    echo "start $watermark_type"
    for model in "${models[@]}"; do
        for token in  "${tokens[@]}"; do

        # dipper need 48gb memory
        for dipper_lex in "${dipper_lexs[@]}"; do
           CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name dipper-attack-"$watermark_type"-hc3 --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/hc3/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/hc3/"$model"/dipper_l"$dipper_lex"_o0 --attack_method dipper --lex "$dipper_lex" 
        done
      
      
        echo "finish $token"
         done
        
        echo "finish $model"
    done
    echo "finish $watermark_type"
done
echo "finish all"
