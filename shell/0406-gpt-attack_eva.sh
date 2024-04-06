#!/bin/bash
#===============================================================================
# usecommand nohup bash shell/0406-gpt-attack_eva.sh > shell/0406-gpt-attack_eva.log 2>&1 &
#===============================================================================

# SUPPORTED_ATTACK_METHODS = [ "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym"]
# watermark_types=("aiwei23b")
watermark_types=("john23" "xuandong23b" "aiwei23b"		"rohith23"	"scott22"			"xiaoniu23" "aiwei23")


models=("llama")

# ===============================================================================


gpus=("1,2,3")

attack_times=("gptattack_5")
for attack_time in "${attack_times[@]}"; do
echo "start $attack_time"

  for watermark_type in "${watermark_types[@]}"; do
      echo "start $watermark_type"
      for model in "${models[@]}"; do
        

                # 对每个元素执行CUDA_VISIBLE_DEVICES=0,2 python命令
                CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/evaluation_pipeline.py --only_attack_zscore True   --overwrite_output_file True --watermark "$watermark_type" --run_name gpt-"$attack_time"  --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/token_200/"$watermark_type"/c4/"$model"/"$attack_time" --wandb_project eva-llama-attack --overwrite_args True
    

          
      done
      echo "finish $watermark_type"

  done
echo "finish $attack_time"
done

echo "finish all"
