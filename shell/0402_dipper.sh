#!/bin/bash
#===============================================================================
# usecommand  nohup bash shell/0402_dipper.sh > shell/0402_dipper.log 2>&1 &
#===============================================================================


watermark_types=("lean23" "aiwei23b" "scott22") 


models=("llama")
attack_types=("swap" "translation")
synonym_probs=("0.4")
helm_attack_methods=("MisspellingAttack" "TypoAttack" "ContractionAttack" "LowercaseAttack" "ExpansionAttack")
dipper_lexs=("20" "40")
cp_attack_insertion_lens=("10")
gpus=("3")
tokens=("token_200")

for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        for token in  "${tokens[@]}"; do
    echo "start $token"
       
  # dipper need 48gb memory
        for dipper_lex in "${dipper_lexs[@]}"; do
           CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0 --attack_method dipper --lex "$dipper_lex" 
        done
        # dipper need 48gb memory
        for dipper_lex in "${dipper_lexs[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-c4-"$model"-dipper_l"$dipper_lex"_o0   --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0   --wandb_project eva-dipper-attack --overwrite_args True
        done

    done
    done
    echo "finish $model"
done
echo "finish $watermark_type"

