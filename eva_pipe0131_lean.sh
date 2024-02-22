#!/bin/bash
#===============================================================================
# usecommand  nohup bash eva_pipe0131_lean.sh > eva_pipe_lean23_23gpu.log 2>&1 &
#===============================================================================

# 定义一个包含不同watermark类型的数组
# SUPPORTED_ATTACK_METHODS = [ "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym"]
watermark_types=("john23") 
# watermark_types=("john23"	"xuandong23b"	"rohith23"	"lean23"	"aiwei23"		"xiaoniu23")
#gpu1
# watermark_types=(	"xuandong23b"	"kiyoon23"	"xiaoniu23")
#gpu2
# watermark_types=(	"lean23" "aiwei23" "rohith23")

models=("opt")
# attack_types=( "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym")
attack_types=( "swap")
synonym_probs=( "0.4" )
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" )
cp_attack_insertion_lens=("10" )
# cp_attack_types=("single-single" "triple-single")
# 遍历数组中的每个元素
# CUDA_VISIBLE_DEVICES=0,2,0  CUDA_VISIBLE_DEVICES=0,2 python watermark_reliability_release/evaluation_pipeline.py --wandb True --only_attack_zscore True --evaluation_metrics z-score  --overwrite_output_file True --watermark aiwei23 --only_attack_zscore True --evaluation_metrics z-score  --overwrite_output_file True --watermark "$watermark_type" --run_name eva-aiwei23-hc3-dipper --input_dir /home/jkl6486/sok-llm-watermark/runs/aiwei23/hc3/dipper --evaluation_metrics all --wandb_project gen-c4-v2.0-1000  --limit_rows 500
for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        # # swap
        for attack_type in "${attack_types[@]}"; do
            # 对每个元素执行CUDA_VISIBLE_DEVICES=0,2 python命令
            CUDA_VISIBLE_DEVICES=2,3 python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --evaluation_metrics z-score  --overwrite_output_file True --watermark "$watermark_type" --run_name "$attack_type"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/"$attack_type" --wandb_project gen-c4-v2.0-1000  --limit_rows 500
        done


        #synonym
        for synonym_prob in "${synonym_probs[@]}"; do
            CUDA_VISIBLE_DEVICES=2,3 python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --evaluation_metrics z-score  --overwrite_output_file True --watermark "$watermark_type" --run_name synonym-"$synonym_prob"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/synonym-"$synonym_prob" --wandb_project gen-c4-v2.0-1000  --limit_rows 500
        done

        # copypaste
        for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
            CUDA_VISIBLE_DEVICES=2,3 python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --evaluation_metrics z-score  --overwrite_output_file True --watermark "$watermark_type" --run_name copypaste-1-"$cp_attack_insertion_len"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/copypaste-1-"$cp_attack_insertion_len"  --wandb_project gen-c4-v2.0-1000  --limit_rows 500
            CUDA_VISIBLE_DEVICES=2,3 python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --evaluation_metrics z-score  --overwrite_output_file True --watermark "$watermark_type" --run_name copypaste-3-"$cp_attack_insertion_len"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/copypaste-3-"$cp_attack_insertion_len" --wandb_project gen-c4-v2.0-1000  --limit_rows 500
        done

        #helm
        for helm_attack_method in "${helm_attack_methods[@]}"; do
            CUDA_VISIBLE_DEVICES=2,3 python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --evaluation_metrics z-score  --overwrite_output_file True --watermark "$watermark_type" --run_name helm-"$helm_attack_method"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/"$helm_attack_method"  --wandb_project gen-c4-v2.0-1000  --limit_rows 500
        done

        dipper need 48gb memory
        for dipper_lex in "${dipper_lexs[@]}"; do
            CUDA_VISIBLE_DEVICES=2,3 python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --evaluation_metrics z-score  --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-c4-"$model"-dipper_l"$dipper_lex"_o0   --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0   --wandb_project gen-c4-v2.0-1000  --limit_rows 500
        done
        # CUDA_VISIBLE_DEVICES=0,2 python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True --evaluation_metrics z-score  --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-c4-"$model"-dipper_l60_o20   --input_dir /home/jkl6486/sok-llm-watermark/runs/"$watermark_type"/c4/"$model"/dipper_l60_o20  --wandb_project gen-c4-v2.0-1000  --limit_rows 500

    done
    echo "finish $model"
done
echo "finish $watermark_type"


