# !/bin/bash
# ===============================================================================
# usecommand nohup bash shell/0331_attack.sh > shell/0331_attack.log 2>&1 &
# ===============================================================================
echo "Current date and time: $(date)"
# ===============================================================================




# models=("opt" "llama")
gpus=("0")



# # attack
models=( "opt")
# # ===============================================================================
attack_types=( "translation")
synonym_probs=( "0.4" )
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" "40")
cp_attack_insertion_lens=("10" "25")
tokens=("token_200")


# echo "Current date and time: $(date)"
# # 遍历数组中的每个元素



# #eva
watermark_types=("aiwei23b" )
for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
    for token in  "${tokens[@]}"; do
    echo "start $token"
        # swap
        for attack_type in "${attack_types[@]}"; do
            # 对每个元素执行CUDA_VISIBLE_DEVICES=0,2 python命令
            CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name "$attack_type"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/"$attack_type" --attack_method "$attack_type"

            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --run_name "$attack_type"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/"$attack_type" --wandb_project c4-eva-attack --overwrite_args True
        done
  
    done
    done
    echo "finish $model"
done
echo "finish $watermark_type"

echo "Current date and time: $(date)"

# #eva
watermark_types=("lean23" )
for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
    for token in  "${tokens[@]}"; do
    echo "start $token"
        # swap
            for dipper_lex in "${dipper_lexs[@]}"; do


           CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0 --attack_method dipper --lex "$dipper_lex" 


            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-c4-"$model"-dipper_l"$dipper_lex"_o0   --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0   --wandb_project eva-aiwei23b-attack --overwrite_args True
        done
  
    done
    done
    echo "finish $model"
done
echo "finish $watermark_type"