#!/bin/bash
#===============================================================================
# usecommand nohup bash shell/gen_pipeline_aiwei23b_all.sh > shell/gen_pipeline_aiwei23b_all.log 2>&1 &
#===============================================================================

watermark_types=("aiwei23b")

models=("opt")
gpus=("0,2,3")

for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"s
        
        # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpus" python命令
        # CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project aiwei23b_0328 --run_name gen-c4-"$watermark_type"-"$model"  --output_dir token_200/"$watermark_type"/c4/"$model" --min_generations 1000 --max_new_tokens 200 --use_sampling False --generation_batch_size 1 --num_beams 1 --model_name_or_path facebook/opt-1.3b
        CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project aiwei23b_0328 --run_name eva-c4-"$watermark_type"-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model" --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/temp --overwrite_args True 
    echo "finish $model"

    done
echo "finish $watermark_type"
done



# watermark_types=("aiwei23b")
# gpus=("1,2,3")

# models=("llama")

# for watermark_type in "${watermark_types[@]}"; do
# echo "start $watermark_type"
#     for model in "${models[@]}"; do
#     echo "start $model"s
        
#         # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpus" python命令
#         CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project aiwei23b_0328 --dataset_name c4 --run_name gen-c4-"$watermark_type"-"$model"  --output_dir token_200/"$watermark_type"/c4/"$model" --min_generations 700 --max_new_tokens 200 --use_sampling False --generation_batch_size 1 --num_beams 1 --model_name_or_path meta-llama/Llama-2-7b-chat-hf
#         CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project aiwei23b_0328 --run_name eva-c4-"$watermark_type"-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model" --overwrite_args True 
#     echo "finish $model"

#     done
# echo "finish $watermark_type"
# done


# #=====hc3

# watermark_types=("aiwei23b")
# gpus=("1,2,3")

# models=("opt")

# for watermark_type in "${watermark_types[@]}"; do
# echo "start $watermark_type"
#     for model in "${models[@]}"; do
#     echo "start $model"
        
#         CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project aiwei23b_0328 --dataset_name hc3 --run_name gen-hc3-"$watermark_type"-"$model"  --output_dir token_200/"$watermark_type"/hc3/"$model" --min_generations 700 --max_new_tokens 200 --use_sampling False --generation_batch_size 1 --num_beams 1 --model_name_or_path facebook/opt-1.3b 
#         CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project aiwei23b_0328 --run_name eva-hc3-"$watermark_type"-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model" --overwrite_args True 
#     echo "finish $model"

#     done
# echo "finish $watermark_type"
# done
# echo "Current date and time: $(date)"

# watermark_types=("aiwei23b")
# gpus=("1,2,3")

# models=("llama")

# for watermark_type in "${watermark_types[@]}"; do
# echo "start $watermark_type"
#     for model in "${models[@]}"; do
#     echo "start $model"s
        
#         CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/generation_pipeline.py --overwrite True --watermark "$watermark_type" --wandb_project aiwei23b_0328 --dataset_name hc3 --run_name gen-hc3-"$watermark_type"-"$model"  --output_dir token_200/"$watermark_type"/hc3/"$model" --min_generations 700 --max_new_tokens 200 --use_sampling False --generation_batch_size 1 --num_beams 1 --model_name_or_path meta-llama/Llama-2-7b-chat-hf
#         CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --overwrite_output_file True --watermark "$watermark_type" --wandb_project aiwei23b_0328 --run_name eva-hc3-"$watermark_type"-"$model"  --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model" --overwrite_args True 
#     echo "finish $model"

#     done
# echo "finish $watermark_type"
# done




# watermark_types=( "aiwei23b")
# models=("llama" "opt")
# # ===============================================================================
# attack_types=( "swap" "translation")
# synonym_probs=( "0.4" )
# helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
# dipper_lexs=("20" "40")
# cp_attack_insertion_lens=("10" "25")
# gpu=("3,2,1")
# # ===============================================================================

# echo "Current date and time: $(date)"
# # 遍历数组中的每个元素
# for watermark_type in "${watermark_types[@]}"; do
#     echo "start $watermark_type"
#     for model in "${models[@]}"; do
#             # dipper need 48gb memory
#         for dipper_lex in "${dipper_lexs[@]}"; do

#             CUDA_VISIBLE_DEVICES="$gpu" CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0 --attack_method dipper --lex "$dipper_lex" 
#         done

#          for attack_type in "${attack_types[@]}"; do
#              # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpu" python 命令
#              # CUDA_VISIBLE_DEVICES="$gpu" python  main.py --watermark "$attack_type"
#              CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name "$attack_type"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/"$attack_type" --attack_method "$attack_type"
#          done


#         # # #synonym
#          for synonym_prob in "${synonym_probs[@]}"; do
#              CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name synonym-"$synonym_prob"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/synonym-"$synonym_prob" --attack_method synonym --synonym_p "$synonym_prob"
#          done

#         # # copypaste
#         for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
#             CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --cp_attack_type single-single --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name copypaste-1-"$cp_attack_insertion_len"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/copypaste-1-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"
#             # if [ $cp_attack_insertion_len != "25" ]; then
#             CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --cp_attack_type triple-single --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name copypaste-3-"$cp_attack_insertion_len"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/copypaste-3-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"
#             # fi
#         done

#         # #helm
#         for helm_attack_method in "${helm_attack_methods[@]}"; do
#             CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name helm-"$helm_attack_method"-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/"$helm_attack_method" --attack_method helm --helm_attack_method "$helm_attack_method"
#         done


#         # CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project c4-attack --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/c4/"$model"/dipper_l60_o20 --attack_method dipper --lex 60 --order 20
#         echo "finish $model"
#     done
#     echo "finish $watermark_type"
# done 
# echo "finish all"
# echo "Current date and time: $(date)"



# watermark_types=( "aiwei23b")
# models=("llama" "opt")
# # ===============================================================================
# attack_types=( "swap" "translation")
# synonym_probs=( "0.4" )
# helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
# dipper_lexs=("20" "40")
# cp_attack_insertion_lens=("10" "25")
# gpu=("3,2,1")
# # ===============================================================================

# echo "Current date and time: $(date)"
# # 遍历数组中的每个元素
# for watermark_type in "${watermark_types[@]}"; do
#     echo "start $watermark_type"
#     for model in "${models[@]}"; do
#             # dipper need 48gb memory
#         for dipper_lex in "${dipper_lexs[@]}"; do

#             CUDA_VISIBLE_DEVICES="$gpu" CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project hc3-attack --run_name dipper-attack-"$watermark_type"-hc3 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model"/dipper_l"$dipper_lex"_o0 --attack_method dipper --lex "$dipper_lex" 
#         done

#          for attack_type in "${attack_types[@]}"; do
#              # 对每个元素执行CUDA_VISIBLE_DEVICES="$gpu" python 命令
#              # CUDA_VISIBLE_DEVICES="$gpu" python  main.py --watermark "$attack_type"
#              CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project hc3-attack --run_name "$attack_type"-attack-"$watermark_type"-hc3 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model"/"$attack_type" --attack_method "$attack_type"
#          done


#         # # #synonym
#          for synonym_prob in "${synonym_probs[@]}"; do
#              CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project hc3-attack --run_name synonym-"$synonym_prob"-attack-"$watermark_type"-hc3 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model"/synonym-"$synonym_prob" --attack_method synonym --synonym_p "$synonym_prob"
#          done

#         # # copypaste
#         for cp_attack_insertion_len in "${cp_attack_insertion_lens[@]}"; do
#             CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --cp_attack_type single-single --overwrite_args True --overwrite_output_file True --wandb_project hc3-attack --run_name copypaste-1-"$cp_attack_insertion_len"-attack-"$watermark_type"-hc3 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model"/copypaste-1-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"
#             # if [ $cp_attack_insertion_len != "25" ]; then
#             CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --cp_attack_type triple-single --overwrite_args True --overwrite_output_file True --wandb_project hc3-attack --run_name copypaste-3-"$cp_attack_insertion_len"-attack-"$watermark_type"-hc3 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model"/copypaste-3-"$cp_attack_insertion_len" --attack_method copy-paste --cp_attack_insertion_len "$cp_attack_insertion_len"
#             # fi
#         done

#         # #helm
#         for helm_attack_method in "${helm_attack_methods[@]}"; do
#             CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project hc3-attack --run_name helm-"$helm_attack_method"-attack-"$watermark_type"-hc3 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model"/"$helm_attack_method" --attack_method helm --helm_attack_method "$helm_attack_method"
#         done


#         # CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES="$gpu" python  watermark_reliability_release/attack_pipeline.py --overwrite_args True --overwrite_output_file True --wandb_project hc3-attack --run_name dipper-attack-"$watermark_type"-hc3 --input_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/token_200/"$watermark_type"/hc3/"$model"/dipper_l60_o20 --attack_method dipper --lex 60 --order 20
#         echo "finish $model"
#     done
#     echo "finish $watermark_type"
# done 
# echo "finish all"
# echo "Current date and time: $(date)"

