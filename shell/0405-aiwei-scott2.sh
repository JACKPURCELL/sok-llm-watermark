#!/bin/bash
#===============================================================================
# usecommand  nohup bash shell/0405-aiwei-scott2.sh > shell/0405-aiwei-scott2.log 2>&1 &
#===============================================================================

# 定义一个包含不同watermark类型的数组
# SUPPORTED_ATTACK_METHODS = [ "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym"]
watermark_types=("aiwei23b") 
# watermark_types=("john23"	"xuandong23b"	"rohith23"	"llama"	"aiwei23"		"xiaoniu23")
#gpu1
# watermark_types=(	"xuandong23b"	"kiyoon23"	"xiaoniu23")
#gpu2
# watermark_types=(	"llama" "aiwei23" "rohith23")

models=("llama")
# attack_types=( "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym")
attack_types=(  "translation")
synonym_probs=( "0.4" )
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" "40")
cp_attack_insertion_lens=("10" "25")
gpus=("0,1")
tokens=("token_200")
# cp_attack_types=("single-single" "triple-single")
# 遍历数组中的每个元素
# CUDA_VISIBLE_DEVICES=0,2,0  CUDA_VISIBLE_DEVICES=0,2 python watermark_reliability_release/evaluation_pipeline.py --wandb True --only_attack_zscore True   --overwrite_output_file True --watermark aiwei23 --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name eva-aiwei23-c4-dipper --input_dir /home/jkl6486/sok-llm-watermark/runs_server3_server3/"$token"/aiwei23/c4/dipper --evaluation_metrics all --wandb_project eva-llama-attack --overwrite_args True
for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        for token in  "${tokens[@]}"; do
    echo "start $token"



        # for dipper_lex in "${dipper_lexs[@]}"; do
        #    CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0 --attack_method dipper --lex "$dipper_lex" 
        # done
      



        # dipper need 48gb memory
        for dipper_lex in "${dipper_lexs[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-c4-"$model"-dipper_l"$dipper_lex"_o0   --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0   --wandb_project eva-llama-attack --overwrite_args True
        done

    done
    done
    echo "finish $model"
done
echo "finish $watermark_type"





# watermark_types=("scott22") 

# for watermark_type in "${watermark_types[@]}"; do
# echo "start $watermark_type"
#     for model in "${models[@]}"; do
#     echo "start $model"
#         for token in  "${tokens[@]}"; do
#     echo "start $token"


    


#         # for dipper_lex in "${dipper_lexs[@]}"; do
#         #    CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0 --attack_method dipper --lex "$dipper_lex" 
#         # done
      


#         # dipper need 48gb memory
#         for dipper_lex in "${dipper_lexs[@]}"; do
#             CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-c4-"$model"-dipper_l"$dipper_lex"_o0   --input_dir /home/jkl6486/sok-llm-watermark/runs_server3_server3/"$token"/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0   --wandb_project eva-llama-attack --overwrite_args True
#         done

#     done
#     done
#     echo "finish $model"
# done
# echo "finish $watermark_type"


