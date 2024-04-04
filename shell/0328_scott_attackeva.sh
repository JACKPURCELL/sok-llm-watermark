#!/bin/bash
#===============================================================================
# usecommand nohup bash shell/0328_scott_attackeva.sh > shell/0328_scott_attackeva2.log 2>&1 &
#===============================================================================

# SUPPORTED_ATTACK_METHODS = [ "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym"]
# watermark_types=( "aiwei23")
# watermark_types=("john23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23" "aiwei23b" "lean23")
# watermark_types=( "xuandong23b" "aiwei23" "aiwei23b" "lean23" )
watermark_types=(	"scott22")
# watermark_types=("john23"	"xuandong23b"	"rohith23"		"aiwei23"		"xiaoniu23")
# attack_types=( "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym")
# ===============================================================================

synonym_probs=( "0.4" )
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" "40")
cp_attack_insertion_lens=("10" "25")
gpu=("2")
gpus=("2")



#====
models=("opt")


# echo "Current date and time: $(date)"
# # 遍历数组中的每个元素
# for watermark_type in "${watermark_types[@]}"; do
#     echo "start $watermark_type"
#     for model in "${models[@]}"; do
         
#          for attack_type in "${attack_types[@]}"; do
#              # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpu" python 命令
#              # CUDA_VISIBLE_DEVICES="$gpu" python  main.py --watermark "$attack_type"
#              CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name "$attack_type"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/"$attack_type" --attack_method "$attack_type"
#          done

#         echo "finish $model"
#     done
#     echo "finish $watermark_type"
# done
# echo "finish all"
# echo "Current date and time: $(date)"

#====
attack_types=( "swap" "translation")

for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        for token in  "${tokens[@]}"; do
    echo "start $token"
        # swap
        for attack_type in "${attack_types[@]}"; do
            # 对每个元素执行CUDA_VISIBLE_DEVICES=0,2 python命令
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name "$attack_type"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/"$attack_type" --wandb_project eva-aiwei23b-attack --overwrite_args True
        done


        #synonym
        for synonym_prob in "${synonym_probs[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name synonym-"$synonym_prob"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/synonym-"$synonym_prob" --wandb_project eva-aiwei23b-attack --overwrite_args True
        done

        # copypaste
        for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --overwrite_output_file True --watermark "$watermark_type" --run_name copypaste-1-"$cp_attack_insertion_len"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/copypaste-1-"$cp_attack_insertion_len"  --wandb_project temporary --overwrite_args True
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --overwrite_output_file True --watermark "$watermark_type" --run_name copypaste-3-"$cp_attack_insertion_len"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/copypaste-3-"$cp_attack_insertion_len" --wandb_project temporary --overwrite_args True
        done

        #helm
        for helm_attack_method in "${helm_attack_methods[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name helm-"$helm_attack_method"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/"$helm_attack_method"  --wandb_project eva-aiwei23b-attack --overwrite_args True
        done

        #dipper need 48gb memory
        for dipper_lex in "${dipper_lexs[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-c4-"$model"-dipper_l"$dipper_lex"_o0   --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0   --wandb_project eva-aiwei23b-attack --overwrite_args True
        done
        # CUDA_VISIBLE_DEVICES=0,2 python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-c4-"$model"-dipper_l60_o20   --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/dipper_l60_o20  --wandb_project eva-aiwei23b-attack --overwrite_args True
    done
    done
    echo "finish $model"
done
echo "finish $watermark_type"



# ===============================================================================
# attack_types=( "swap")
# synonym_probs=("0.1" "0.2" "0.4" "0.8" "1.0")
# helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
# dipper_lexs=("20" "40" "60")
# cp_attack_insertion_lens=("10" "25")
# ===============================================================================
# cp_attack_types=("single-single" "triple-single")
models=("llama")

echo "Current date and time: $(date)"
# 遍历数组中的每个元素
for watermark_type in "${watermark_types[@]}"; do
    echo "start $watermark_type"
    for model in "${models[@]}"; do
            # dipper need 48gb memory
#         for dipper_lex in "${dipper_lexs[@]}"; do
# echo "Current date and time: $(date)"

#             CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0 --attack_method dipper --lex "$dipper_lex" 
#         done

         for attack_type in "${attack_types[@]}"; do
             # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpu" python 命令
             # CUDA_VISIBLE_DEVICES="$gpu" python  main.py --watermark "$attack_type"
             CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name "$attack_type"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/"$attack_type" --attack_method "$attack_type"
         done


        # #synonym
         for synonym_prob in "${synonym_probs[@]}"; do
             CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name synonym-"$synonym_prob"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/synonym-"$synonym_prob" --attack_method synonym --synonym_p "$synonym_prob"
         done

        # copypaste
        for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --cp_attack_type single-single --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name copypaste-1-"$cp_attack_insertion_len"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/copypaste-1-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"
            # if [ $cp_attack_insertion_len != "25" ]; then
            CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --cp_attack_type triple-single --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name copypaste-3-"$cp_attack_insertion_len"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/copypaste-3-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"
            # fi
        done

        #helm
        for helm_attack_method in "${helm_attack_methods[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name helm-"$helm_attack_method"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/"$helm_attack_method" --attack_method helm --helm_attack_method "$helm_attack_method"
        done


        # CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/dipper_l60_o20 --attack_method dipper --lex 60 --order 20
        echo "finish $model"
    done
    echo "finish $watermark_type"
done
echo "finish all"
echo "Current date and time: $(date)"



# =======
for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        for token in  "${tokens[@]}"; do
    echo "start $token"
        # swap
        for attack_type in "${attack_types[@]}"; do
            # 对每个元素执行CUDA_VISIBLE_DEVICES=0,2 python命令
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name "$attack_type"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/"$attack_type" --wandb_project eva-aiwei23b-attack --overwrite_args True
        done


        #synonym
        for synonym_prob in "${synonym_probs[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name synonym-"$synonym_prob"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/synonym-"$synonym_prob" --wandb_project eva-aiwei23b-attack --overwrite_args True
        done

        # copypaste
        for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --overwrite_output_file True --watermark "$watermark_type" --run_name copypaste-1-"$cp_attack_insertion_len"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/copypaste-1-"$cp_attack_insertion_len"  --wandb_project temporary --overwrite_args True
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --overwrite_output_file True --watermark "$watermark_type" --run_name copypaste-3-"$cp_attack_insertion_len"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/copypaste-3-"$cp_attack_insertion_len" --wandb_project temporary --overwrite_args True
        done

        #helm
        for helm_attack_method in "${helm_attack_methods[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name helm-"$helm_attack_method"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/"$helm_attack_method"  --wandb_project eva-aiwei23b-attack --overwrite_args True
        done

        #dipper need 48gb memory
        # for dipper_lex in "${dipper_lexs[@]}"; do
        #     CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-c4-"$model"-dipper_l"$dipper_lex"_o0   --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0   --wandb_project eva-aiwei23b-attack --overwrite_args True
        # done
        # CUDA_VISIBLE_DEVICES=0,2 python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-c4-"$model"-dipper_l60_o20   --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/dipper_l60_o20  --wandb_project eva-aiwei23b-attack --overwrite_args True
    done
    done
    echo "finish $model"
done
echo "finish $watermark_type"


