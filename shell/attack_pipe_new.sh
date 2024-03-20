#!/bin/bash
#===============================================================================
# usecommand nohup bash shell/attack_pipe_new.sh > shell/attack_pipe_others.log 2>&1 &
#===============================================================================

# SUPPORTED_ATTACK_METHODS = [ "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym"]
watermark_types=("xiaoniu23")
# "xiaoniu23" "xuandong23b" "aiwei23" "aiwei23b"
#"john23" "lean23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23" "kiyoon23" "|"
models=("opt")

# attack_types=( "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym")
# ===============================================================================
attack_types=( "swap")
synonym_probs=( "0.4" )
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("40")
cp_attack_insertion_lens=("25")
gpus=("2")
# ===============================================================================
# attack_types=( "swap")
# synonym_probs=("0.1" "0.2" "0.4" "0.8" "1.0")
# helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
# dipper_lexs=("20" "40" "60")
#cp_attack_insertion_lens=("25")
# ===============================================================================
# cp_attack_types=("single-single" "triple-single")
# 遍历数组中的每个元素
for watermark_type in "${watermark_types[@]}"; do
    echo "start $watermark_type"
    for model in "${models[@]}"; do

        #  for attack_type in "${attack_types[@]}"; do
        #     # 对每个元素执行python命令
        #     python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name "$attack_type"-attack-"$watermark_type"-c4 --input_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"-"$beams"beams  --output_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"-"$beams"beams/"$attack_type" --attack_method "$attack_type"
        #  done


        # #synonym
        #  for synonym_prob in "${synonym_probs[@]}"; do
        #     python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name synonym-"$synonym_prob"-attack-"$watermark_type"-c4 --input_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"-"$beams"beams  --output_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"-"$beams"beams/synonym-"$synonym_prob" --attack_method synonym --synonym_p "$synonym_prob"
        #  done

        #copypaste
        # for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
        #     python watermark_reliability_release/attack_pipeline.py --cp_attack_type single-single --wandb False --overwrite_args True  --run_name copypaste-1-"$cp_attack_insertion_len"-attack-"$watermark_type"-c4 --input_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/copypaste-1-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"
        #     #if [$cp_attack_insertion_len != "25"]; then
        #     python watermark_reliability_release/attack_pipeline.py --cp_attack_type triple-single --wandb False --overwrite_args True  --run_name copypaste-3-"$cp_attack_insertion_len"-attack-"$watermark_type"-c4 --input_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/copypaste-3-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"
        #     #fi
        # done

        # # helm
        # for helm_attack_method in "${helm_attack_methods[@]}"; do
        #    python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name helm-"$helm_attack_method"-attack-"$watermark_type"-c4 --input_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"-"$beams"beams  --output_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"-"$beams"beams/"$helm_attack_method" --attack_method helm --helm_attack_method "$helm_attack_method"
        # done

        # dipper need 48gb memory
        for dipper_lex in "${dipper_lexs[@]}"; do
           CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0 --attack_method dipper --lex "$dipper_lex" 
        done
        #CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"-"$beams"beams  --output_dir /home/ljc/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"-"$beams"beams/dipper_l60_o20 --attack_method dipper --lex 60 --order 20
      
        
        echo "finish $model"
    done
    echo "finish $watermark_type"
done
echo "finish all"

