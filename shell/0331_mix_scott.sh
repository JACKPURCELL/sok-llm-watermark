# !/bin/bash
# ===============================================================================
# usecommand nohup bash shell/0331_mix_scott.sh > shell/0331_mix_scott_eva.log 2>&1 &
# ===============================================================================
echo "Current date and time: $(date)"
# ===============================================================================

watermark_types=("scott22" )


# models=("opt" "llama")
gpus=("3")

# for watermark_type in "${watermark_types[@]}"; do
# echo "start $watermark_type"
#     for model in "${models[@]}"; do
#     echo "start $model"s
        
#         CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project c4 --dataset_name c4 --run_name gen-c4-"$watermark_type"-"$model"  --output_dir token_200/"$watermark_type"/c4/"$model" --min_generations 700 --max_new_tokens 200 --use_sampling False --generation_batch_size 1 --num_beams 1 --model_name_or_path meta-llama/Llama-2-7b-chat-hf
#         CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project c4 --run_name eva-c4-"$watermark_type"-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model" --overwrite_args True 
#     echo "finish $model"

#     done
# echo "finish $watermark_type"
# done





# # attack
models=( "llama")
# # ===============================================================================
attack_types=( "swap" "translation")
synonym_probs=( "0.4" )
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" "40")
cp_attack_insertion_lens=("10" "25")


# echo "Current date and time: $(date)"
# # 遍历数组中的每个元素



# #eva
tokens=("token_200")
for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
    for token in  "${tokens[@]}"; do
    echo "start $token"
        # swap
        for attack_type in "${attack_types[@]}"; do
            # 对每个元素执行CUDA_VISIBLE_DEVICES=0,2 python命令
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --run_name "$attack_type"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/"$attack_type" --wandb_project c4-eva-attack --overwrite_args True
        done


        #synonym
        for synonym_prob in "${synonym_probs[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --run_name synonym-"$synonym_prob"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/synonym-"$synonym_prob" --wandb_project c4-eva-attack --overwrite_args True
        done

        #copypaste
        for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --watermark "$watermark_type" --run_name copypaste-1-"$cp_attack_insertion_len"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/copypaste-1-"$cp_attack_insertion_len"  --wandb_project c4-eva-attack  --overwrite_args True --overwrite_output_file True
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --watermark "$watermark_type" --run_name copypaste-3-"$cp_attack_insertion_len"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/copypaste-3-"$cp_attack_insertion_len" --wandb_project c4-eva-attack  --overwrite_args True --overwrite_output_file True
        done

        #helm
        for helm_attack_method in "${helm_attack_methods[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --run_name helm-"$helm_attack_method"-neweva-"$watermark_type"-c4-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/"$helm_attack_method"  --wandb_project c4-eva-attack --overwrite_args True 
        done

  
    done
    done
    echo "finish $model"
done
echo "finish $watermark_type"

echo "Current date and time: $(date)"

# # sleep 3h
# echo "Current date and time: $(date)"

# for watermark_type in "${watermark_types[@]}"; do
#     echo "start $watermark_type"
#     for model in "${models[@]}"; do

#         for dipper_lex in "${dipper_lexs[@]}"; do

#             CUDA_VISIBLE_DEVICES="$gpus" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0 --attack_method dipper --lex "$dipper_lex" 
#         done

#                # # dipper need 48gb memory
#         for dipper_lex in "${dipper_lexs[@]}"; do
#             CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-c4-"$model"-dipper_l"$dipper_lex"_o0   --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0   --wandb_project c4-eva-attack --overwrite_args True 
#         done

#         echo "finish $model"
#     done
#     echo "finish $watermark_type"
# done 
# echo "finish all"
# echo "Current date and time: $(date)"


