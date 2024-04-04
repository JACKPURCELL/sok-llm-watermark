#!/bin/bash
#===============================================================================
# usecommand  nohup bash shell/0401_scott_dp.sh > shell/0401_scott_dp.log 2>&1 &
#===============================================================================


watermark_types=("scott22") 


models=("opt")
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
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-c4-"$model"-dipper_l"$dipper_lex"_o0   --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o0   --wandb_project eva-scott22-attack --overwrite_args True
        done

    done
    done
    echo "finish $model"
done
echo "finish $watermark_type"


