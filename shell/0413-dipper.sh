#!/bin/bash
#===============================================================================
# usecommand  nohup bash shell/0413-dipper.sh > shell/0422-dipper-rep51234.log 2>&1 &
#===============================================================================

# 定义一个包含不同watermark类型的数组
# SUPPORTED_ATTACK_METHODS = [ "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym"]
watermark_types=("john23" "xuandong23b" "rohith23" "scott22")
# watermark_types=( "aiwei23b"		"rohith23") 
# watermark_types=(	"scott22"	"aiwei23"		"xiaoniu23")
# watermark_types=("john23"	"xuandong23b"	"rohith23"	"aiwei23b"	"aiwei23"		"scott22" "xiaoniu23")
#gpu1
# watermark_types=(	"xuandong23b"	"kiyoon23"	"xiaoniu23")
#gpu2
# watermark_types=(	"llama" "aiwei23" "rohith23")

models=("opt")
# attack_types=( "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym")

dipper_lexs=( "40")
dipper_orders=("2" "3" "4")
gpus=("0")
tokens=("token_200")

for watermark_type in "${watermark_types[@]}"; do
echo "start $watermark_type"
    for model in "${models[@]}"; do
    echo "start $model"
        for token in  "${tokens[@]}"; do
    echo "start $token"



        # for dipper_lex in "${dipper_lexs[@]}"; do
        #    CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --run_name dipper-attack-"$watermark_type"-c4 --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"  --output_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/dipper_l"$dipper_lex"_o"$dipper_order" --attack_method dipper --lex "$dipper_lex" --order "$dipper_order"
        # done
      



        # dipper need 48gb memory
        for dipper_lex in "${dipper_lexs[@]}"; do
        for dipper_order in "${dipper_orders[@]}"; do
            CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True  --evaluation_metrics z-score --overwrite_output_file True --watermark "$watermark_type" --run_name dipper-neweva-"$watermark_type"-c4-"$model"-dipper_l"$dipper_lex"_o"$dipper_order"   --input_dir /home/jkl6486/sok-llm-watermark/runs/"$token"/"$watermark_type"/c4/"$model"/dipper_"$dipper_lex"_rep"$dipper_order"   --wandb_project dipperorder --overwrite_args True
        done
        done

    done
    done
    echo "finish $model"
done
echo "finish $watermark_type"



