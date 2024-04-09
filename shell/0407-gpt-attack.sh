#!/bin/bash
#===============================================================================
# usecommand nohup bash shell/0407-gpt-attack.sh > shell/0407-gpt-attack2.log 2>&1 &
#===============================================================================

# SUPPORTED_ATTACK_METHODS = [ "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym"]
# watermark_types=("aiwei23b")
watermark_types=("john23" "xuandong23b" "aiwei23b"		"rohith23"	"scott22"	"aiwei23"		"xiaoniu23")

# john23 unrun
# "xiaoniu23" "xuandong23b" "aiwei23" "aiwei23b"
#"john23" "lean23" "rohith23" "xiaoniu23" "xuandong23b" "aiwei23" "kiyoon23" "|"
models=("llama")
# num_beams=( "4" "8")
dirs=("exp_3_gptattack_base_0" "exp_3_gptattack_1" "exp_3_gptattack_2" "exp_3_gptattack_3" "exp_3_gptattack_4" "exp_3_gptattack_5" )
# dirs=( "exp_3_gptattack_2" "exp_3_gptattack_3" "exp_3_gptattack_4" "exp_3_gptattack_5" )
# attack_types=( "dipper"  "copy-paste"  "scramble" "helm" "oracle" "swap" "synonym")
# ===============================================================================
attack_types=( "gpt")

gpus=("1")
tokens=( "token_200" )
for index in "${!dirs[@]}"; do
    dir=${dirs[$index]}
    next_dir=${dirs[$((index+1))]}  # Access the next element
    echo "Current dir: $dir"
    echo "Next dir: $next_dir"

if [[ "$dir" != "exp_3_gptattack_base_0" ]]; then
    python multi_attack_gpt.py --exp "$dir"

fi
  

  for watermark_type in "${watermark_types[@]}"; do
      echo "start $watermark_type"
      for model in "${models[@]}"; do
          for token in  "${tokens[@]}"; do
            for attack_type in "${attack_types[@]}"; do
              # 对每个元素执行python命令
              CUDA_VISIBLE_DEVICES="$gpus" python watermark_reliability_release/attack_pipeline.py --wandb False --overwrite_args True  --wandb False --input_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$dir"  --output_dir /home/jkl6486/sok-llm-watermark/runs_server3/"$token"/"$watermark_type"/c4/"$model"/"$next_dir" --attack_method "$attack_type" --overwrite_output_file True
            done

            


        
        
          echo "finish $token"
          done
          
          echo "finish $model"
      done
      echo "finish $watermark_type"
  done
  echo "finish all"

done