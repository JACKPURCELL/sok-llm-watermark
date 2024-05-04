# !/bin/bash
# ===============================================================================
# usecommand nohup bash release_shell/eva_john23.sh > release_shell/eva_john23.log 2>&1 &
# ===============================================================================
echo "Current date and time: $(date)"
# ===============================================================================

watermark_types=("john23" )


models=("llama" "opt")
gpus=("0")
datasets=("c4" "hc3")
attack_types=( "swap" "translation")
synonym_probs=( "0.4" )
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" "40")
cp_attack_insertion_lens=("10" "25")




#eva
tokens=("token_200")

for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
    for token in  "${tokens[@]}"; do
    echo "start $token"
    for dataset in "${datasets[@]}"; do
    echo "start $dataset"
        # swap
        for attack_type in "${attack_types[@]}"; do
            # 对每个元素执行CUDA_VISIBLE_DEVICES=0,2 python命令
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --run_name "$attack_type"-neweva-"$watermark_type"-"$dataset"-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"/"$attack_type" --wandb_project "$dataset"-eva-attack --overwrite_args True
        done


        #synonym
        for synonym_prob in "${synonym_probs[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --run_name synonym-"$synonym_prob"-neweva-"$watermark_type"-"$dataset"-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"/synonym-"$synonym_prob" --wandb_project "$dataset"-eva-attack --overwrite_args True
        done

        #copypaste
        for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --watermark "$watermark_type" --run_name copypaste-1-"$cp_attack_insertion_len"-neweva-"$watermark_type"-"$dataset"-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/"$dataset"/"$model"/copypaste-1-"$cp_attack_insertion_len"  --wandb_project "$dataset"-eva-attack  --overwrite_args True --overwrite_output_file True
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --watermark "$watermark_type" --run_name copypaste-3-"$cp_attack_insertion_len"-neweva-"$watermark_type"-"$dataset"-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/"$dataset"/"$model"/copypaste-3-"$cp_attack_insertion_len" --wandb_project "$dataset"-eva-attack  --overwrite_args True --overwrite_output_file True
        done

        #helm
        for helm_attack_method in "${helm_attack_methods[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --run_name helm-"$helm_attack_method"-neweva-"$watermark_type"-"$dataset"-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"/"$helm_attack_method"  --wandb_project "$dataset"-eva-attack --overwrite_args True 
        done

        # dipper need 48gb memory
        for dipper_lex in "${dipper_lexs[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-"$dataset"-"$model"-dipper_l"$dipper_lex"_o0   --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/"$dataset"/"$model"/dipper_l"$dipper_lex"_o0   --wandb_project "$dataset"-eva-attack --overwrite_args True 
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
