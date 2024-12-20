#!/bin/bash
#===============================================================================
# usecommand nohup bash shell/multi_attack.sh > shell/0407_multiattack_ContractionAttack.log 2>&1 &
#===============================================================================

# SUPPORTED_ATTACK_METHODS = [ "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym"]
watermark_types=("john23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23" "aiwei23b" "scott22")
# watermark_types=("john23")
# john23 unrun
# "xiaoniu23" "xuandong23b" "aiwei23" "aiwei23b"
#"john23" "lean23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23" "kiyoon23" "|"
models=("opt")
# attack_types=( "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym")
# ===============================================================================
attack_types=( "swap")
synonym_prob=( "0.4" )
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" "40")
cp_attack_insertion_lens=("10" "25")
gpus=("1")
tokens=( "token_200" )
attack_1="ContractionAttack"
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

            mkdir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$attack_1"_multiple
            cp /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/gen_table_meta.json /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$attack_1"_multiple/

            python /home/jkl6486/sok-llm-watermark/watermark_reliability_release/multiattack_preprocessing.py --input_path /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$attack_1"/gen_table_attacked.jsonl --output_path /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$attack_1"_multiple/gen_table.jsonl

            #swap
            for attack_type in "${attack_types[@]}"; do
             # 对每个元素执行python命令
                python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name "$attack_type"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$attack_1"_multiple  --output_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$attack_1"_multiple/"$attack_type" --attack_method "$attack_type"
            done
            #synonym
            for synonym_prob in "${synonym_probs[@]}"; do
                python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name synonym-"$synonym_prob"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$attack_1"_multiple  --output_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$attack_1"_multiple/synonym-"$synonym_prob" --attack_method synonym --synonym_p "$synonym_prob"
            done
            # #copypaste
            # for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
            #     python watermark_reliability_release/attack_pipeline.py --cp_attack_type single-single --wandb False --overwrite_args True  --run_name copypaste-1-"$cp_attack_insertion_len"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/token_200/"$watermark_type"/c4/"$model"/"$attack_1"_multiple  --output_dir /home/jkl6486/sok-llm-watermark/runs_server3/token_200/"$watermark_type"/c4/"$model"/"$attack_1"_multiple/copypaste-1-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"
        
            #     python watermark_reliability_release/attack_pipeline.py --cp_attack_type triple-single --wandb False --overwrite_args True  --run_name copypaste-3-"$cp_attack_insertion_len"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/token_200/"$watermark_type"/c4/"$model"/"$attack_1"_multiple  --output_dir /home/jkl6486/sok-llm-watermark/runs_server3/token_200/"$watermark_type"/c4/"$model"/"$attack_1"_multiple/copypaste-3-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"
            # done
            # helm
            for helm_attack_method in "${helm_attack_methods[@]}"; do
                python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name helm-"$helm_attack_method"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$attack_1"_multiple  --output_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$attack_1"_multiple/"$helm_attack_method" --attack_method helm --helm_attack_method "$helm_attack_method"
            done
            # dipper need 48gb memory
            # for dipper_lex in "${dipper_lexs[@]}"; do
            # CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$attack_1"_multiple  --output_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$attack_1"_multiple/dipper_l"$dipper_lex"_o0 --attack_method dipper --lex "$dipper_lex" 
            # done
            # CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$attack_1"_multiple  --output_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$attack_1"_multiple/dipper_l60_o20 --attack_method dipper --lex 60 --order 20
        done
    done
    echo "finish $watermark_type"
done
echo "finish all"
