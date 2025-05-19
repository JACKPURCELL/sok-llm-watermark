#!/bin/bash
#===============================================================================
# usecommand nohup bash shell/0331_aiwei23b_eva.sh > shell/0331_aiwei23b_eva.log 2>&1 &
#===============================================================================

watermark_types=(	"aiwei23b")
# ===============================================================================
attack_types=( "swap" "translation")

synonym_probs=( "0.4" )
helm_attack_methods=("ContractionAttack" "LowercaseAttack" "ExpansionAttack")
# helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" "40")
cp_attack_insertion_lens=("10" "25")

gpus=("3")



#====
models=("opt")


tokens=( "token_200")
#====

for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        for token in  "${tokens[@]}"; do
    echo "start $token"


        # #synonym
        # for synonym_prob in "${synonym_probs[@]}"; do
        #     CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name synonym-"$synonym_prob"-neweva-"$watermark_type"-arxiv-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/arxiv/"$model"/synonym-"$synonym_prob" --wandb_project eva-aiwei23b-attack --overwrite_args True
        # done

        # # copypaste
        # for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
        #     CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --overwrite_output_file True --watermark "$watermark_type" --run_name copypaste-1-"$cp_attack_insertion_len"-neweva-"$watermark_type"-arxiv-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/arxiv/"$model"/copypaste-1-"$cp_attack_insertion_len"  --wandb_project temporary --overwrite_args True
        #     CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --overwrite_output_file True --watermark "$watermark_type" --run_name copypaste-3-"$cp_attack_insertion_len"-neweva-"$watermark_type"-arxiv-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/arxiv/"$model"/copypaste-3-"$cp_attack_insertion_len" --wandb_project temporary --overwrite_args True
        # done

        #helm
        for helm_attack_method in "${helm_attack_methods[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name helm-"$helm_attack_method"-neweva-"$watermark_type"-arxiv-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/arxiv/"$model"/"$helm_attack_method"  --wandb_project eva-arxiv-attack --overwrite_args True --evaluation_metrics z-score
        done

        # # swap
        # for attack_type in "${attack_types[@]}"; do
        #     # 对每个元素执行CUDA_VISIBLE_DEVICES=0,2 python命令
        #     CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name "$attack_type"-neweva-"$watermark_type"-arxiv-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/arxiv/"$model"/"$attack_type" --wandb_project eva-aiwei23b-attack --overwrite_args True
        # done


        #dipper need 48gb memory
        for dipper_lex in "${dipper_lexs[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-arxiv-"$model"-dipper_l"$dipper_lex"_o0   --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/arxiv/"$model"/dipper_l"$dipper_lex"_o0   --wandb_project eva-arxiv-attack --overwrite_args True --evaluation_metrics z-score
        done
        # CUDA_VISIBLE_DEVICES=0,2 python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-arxiv-"$model"-dipper_l60_o20   --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/arxiv/"$model"/dipper_l60_o20  --wandb_project eva-aiwei23b-attack --overwrite_args True
    done
    done
    echo "finish $model"
done
echo "finish $watermark_type"


