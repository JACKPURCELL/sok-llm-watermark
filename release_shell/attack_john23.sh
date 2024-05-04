#!/bin/bash
#===============================================================================
# usecommand nohup bash release_shell/attack_john23.sh > release_shell/attack_john23.log 2>&1 &
#===============================================================================
echo "Current date and time: $(date)"


# ===============================================================================
watermark_types=("john23" )


models=("llama" "opt")
gpus=("0,1")
datasets=("c4" "hc3")
attack_types=( "swap" "translation")
synonym_probs=( "0.4" )
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" "40")
cp_attack_insertion_lens=("10" "25")
tokens=("token_200")



echo "Current date and time: $(date)"
# 遍历数组中的每个元素
for watermark_type in "${watermark_types[@]}"; do
    echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
    for token in  "${tokens[@]}"; do
    echo "start $token"
    for dataset in "${datasets[@]}"; do
    echo "start $dataset"

        # #synonym
         for synonym_prob in "${synonym_probs[@]}"; do
             CUDA_VISIBLE_DEVICES="$gpus" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project "$dataset"-attack --run_name synonym-"$synonym_prob"-attack-"$watermark_type"-"$dataset" --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"/synonym-"$synonym_prob" --attack_method synonym --synonym_p "$synonym_prob"
         done

        # copypaste
        for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python  watermark_reliability_release/attack_pipeline.py --cp_attack_type single-single --overwrite_args True --overwrite_output_file True --wandb_project "$dataset"-attack --run_name copypaste-1-"$cp_attack_insertion_len"-attack-"$watermark_type"-"$dataset" --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"/copypaste-1-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"

            CUDA_VISIBLE_DEVICES="$gpus" python  watermark_reliability_release/attack_pipeline.py --cp_attack_type triple-single --overwrite_args True --overwrite_output_file True --wandb_project "$dataset"-attack --run_name copypaste-3-"$cp_attack_insertion_len"-attack-"$watermark_type"-"$dataset" --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"/copypaste-3-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"

        done

        #helm
        for helm_attack_method in "${helm_attack_methods[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project "$dataset"-attack --run_name helm-"$helm_attack_method"-attack-"$watermark_type"-"$dataset" --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"/"$helm_attack_method" --attack_method helm --helm_attack_method "$helm_attack_method"
        done



         for attack_type in "${attack_types[@]}"; do
             # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpus" python 命令
             # CUDA_VISIBLE_DEVICES="$gpus" python  main.py --watermark "$attack_type"
             CUDA_VISIBLE_DEVICES="$gpus" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project "$dataset"-attack --run_name "$attack_type"-attack-"$watermark_type"-"$dataset" --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"/"$attack_type" --attack_method "$attack_type"
         done


        for dipper_lex in "${dipper_lexs[@]}"; do
        echo "Current date and time: $(date)"

            CUDA_VISIBLE_DEVICES="$gpus" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project "$dataset"-attack --run_name dipper-attack-"$watermark_type"-"$dataset" --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"/dipper_l"$dipper_lex"_o0 --attack_method dipper --lex "$dipper_lex" 
        done

    echo "finish $dataset"
    done

    echo "finish $token"
    done

    echo "finish $model"
    done

echo "finish $watermark_type"
done

echo "finish all"
echo "Current date and time: $(date)"


